import hashlib
import json
import logging
import os
import pickle
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import frontmatter
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.cache import SQLiteCache
from langchain_community.retrievers import BM25Retriever
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- 1. Configuration Management ---

logging.basicConfig(level=logging.INFO, filename='myapp.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

class Settings(BaseSettings):
    APP_TITLE: str = "SRC Cluster Knowledge Base API"
    APP_DESCRIPTION: str = "An API to query documentation about Stanford's high-performance computing clusters."
    APP_VERSION: str = "1.0.0"
    MODEL_PATH: str = Field(..., env="MODEL_PATH")
    CLUSTERS: Dict[str, str] = {"sherlock": "sherlock/", "farmshare": "farmshare/", "oak": "oak/", "elm": "elm/"}
    # NEW: Add a configurable cache directory
    DOC_CACHE_DIR: str = "doc_cache/"
    CORS_ORIGINS: List[str] = ['http://localhost:5000', 'http://127.0.0.1:5000']
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# --- 2. Pydantic Models for API ---

class QueryRequest(BaseModel):
    query: str
    cluster: Optional[str] = None

class Source(BaseModel):
    title: str
    url: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    cluster: str
    sources: List[Source]

# --- 3. Core Application Logic (RAG Service) ---

# NEW: A dedicated class to manage document caching
class DocumentCacheManager:
    """Handles caching of processed documents to avoid re-ingestion on every startup."""
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Document cache manager initialized at: {self.cache_dir.resolve()}")

    def _get_cache_paths(self, cluster_name: str) -> Tuple[Path, Path]:
        """Returns the paths for the pickled documents and their state file."""
        state_filename = f"{cluster_name}_state.json"
        docs_filename = f"{cluster_name}_docs.pkl"
        return self.cache_dir / state_filename, self.cache_dir / docs_filename

    def _get_current_file_states(self, corpus_dir: str) -> Dict[str, float]:
        """Gets the current modification times for all .md files in a directory."""
        states = {}
        for filename in os.listdir(corpus_dir):
            if filename.endswith(".md"):
                file_path = os.path.join(corpus_dir, filename)
                states[filename] = os.path.getmtime(file_path)
        return states

    def _is_cache_stale(self, cluster_name: str, corpus_dir: str) -> bool:
        """Checks if the cached documents for a cluster are stale."""
        state_path, docs_path = self._get_cache_paths(cluster_name)

        if not state_path.exists() or not docs_path.exists():
            logger.info(f"Cache miss for '{cluster_name}': No cache files found.")
            return True

        with open(state_path, 'r') as f:
            try:
                cached_states = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Cache miss for '{cluster_name}': Could not decode state file.")
                return True

        current_states = self._get_current_file_states(corpus_dir)
        
        # If the set of filenames or any modification time has changed, cache is stale
        if cached_states != current_states:
            logger.info(f"Cache stale for '{cluster_name}': File states have changed.")
            return True
        
        logger.info(f"Cache hit for '{cluster_name}': File states are unchanged.")
        return False

    def get_documents(self, cluster_name: str, corpus_dir: str) -> List[Document]:
        """
        Main method to get documents for a cluster.
        Loads from cache if available and fresh, otherwise re-ingests and updates cache.
        """
        if self._is_cache_stale(cluster_name, corpus_dir):
            logger.info(f"Re-ingesting documents for cluster '{cluster_name}'...")
            documents = self._ingest_and_cache(cluster_name, corpus_dir)
        else:
            logger.info(f"Loading documents from cache for cluster '{cluster_name}'...")
            state_path, docs_path = self._get_cache_paths(cluster_name)
            try:
                with open(docs_path, 'rb') as f:
                    documents = pickle.load(f)
            except (pickle.UnpicklingError, FileNotFoundError, EOFError) as e:
                logger.warning(f"Failed to load from cache for '{cluster_name}' due to '{e}'. Re-ingesting.")
                documents = self._ingest_and_cache(cluster_name, corpus_dir)
        return documents

    def _ingest_and_cache(self, cluster_name: str, corpus_dir: str) -> List[Document]:
        """Performs the actual file ingestion and then saves the results to the cache."""
        state_path, docs_path = self._get_cache_paths(cluster_name)
        
        # 1. Ingest from source files
        documents = self._ingest_markdown_files(corpus_dir)
        
        # 2. Get current state and save it
        current_states = self._get_current_file_states(corpus_dir)
        with open(state_path, 'w') as f:
            json.dump(current_states, f, indent=2)
            
        # 3. Save the processed documents
        with open(docs_path, 'wb') as f:
            pickle.dump(documents, f)
            
        logger.info(f"Successfully ingested and cached {len(documents)} documents for '{cluster_name}'.")
        return documents

    @staticmethod
    def _ingest_markdown_files(corpus_dir: str) -> List[Document]:
        """Static method to read and parse markdown files into Document objects."""
        documents = []
        for filename in os.listdir(corpus_dir):
            if filename.endswith(".md"):
                file_path = os.path.join(corpus_dir, filename)
                try:
                    post = frontmatter.load(file_path)
                    metadata = post.metadata
                    metadata['source'] = filename
                    doc = Document(page_content=post.content, metadata=metadata)
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Could not read or parse front matter from file {file_path}: {e}")
        return documents

class RAGService:
    def __init__(self, config: Settings):
        self.settings = config
        # NEW: Instantiate the cache manager
        self.doc_cache_manager = DocumentCacheManager(config.DOC_CACHE_DIR)
        self.llm = None
        self.retrievers: Dict[str, BM25Retriever] = {}
        self.chain = None

    # REMOVED: _ingest_markdown_files is now a static method within DocumentCacheManager

    def initialize(self):
        logger.info("Initializing RAG Service...")
        try:
            set_llm_cache(SQLiteCache(database_path=".langchain.db"))
            logger.info("LangChain LLM cache enabled with SQLite.")
        except Exception as e:
            logger.error(f"Could not set up LLM cache: {e}")

        try:
            logger.info(f"Loading model from {self.settings.MODEL_PATH}...")
            model = AutoModelForCausalLM.from_pretrained(
                self.settings.MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(self.settings.MODEL_PATH)
            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=400,
                temperature=0.5, do_sample=True, pad_token_id=tokenizer.eos_token_id,
                return_full_text=False
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("Model and pipeline loaded successfully.")
        except Exception as e:
            logger.critical(f"FATAL: Could not load the model. Error: {e}")
            raise

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        for cluster_name, path in self.settings.CLUSTERS.items():
            if not os.path.isdir(path):
                logger.warning(f"Directory not found for cluster '{cluster_name}': {path}. Skipping.")
                continue
            
            # REFACTORED: Use the cache manager to get documents
            documents = self.doc_cache_manager.get_documents(cluster_name, path)
            
            if not documents:
                logger.warning(f"No documents found or loaded for cluster '{cluster_name}'.")
                continue

            split_docs = text_splitter.split_documents(documents)
            self.retrievers[cluster_name] = BM25Retriever.from_documents(split_docs)
            logger.info(f"Created retriever for '{cluster_name}'.")

        # The RAG chain definition is only needed once, after all retrievers are built.
        # Moved outside the loop for clarity and correctness.
        if not self.retrievers:
            logger.critical("FATAL: No retrievers were created. The application cannot answer queries.")
            # Depending on desired behavior, you might want to raise an exception here.
            return

        prompt_template = ChatPromptTemplate.from_template(
            """<s>[INST] You are a helpful and concise expert assistant for the Stanford Research Computing Center (SRCC).
Your task is to answer the user's query using ONLY the information provided in the CONTEXT below.

- If the CONTEXT does not contain the answer, state that the information is not available in the documentation.
- Be direct. Do not add conversational filler, introductions, or conclusions.

CONTEXT:
{context}

USER QUERY:
{query} [/INST]"""
    )
        prompt_template = ChatPromptTemplate.from_template(
    """<s>[INST] You are a friendly and knowledgeable expert for the Stanford Research Computing Center (SRCC).
Your goal is to provide a helpful answer to the user's query based *only* on the provided documentation context.

- Synthesize the information from the `CONTEXT` into a clear and direct answer.
- If the `CONTEXT` does not contain a direct answer, state that, but also provide any closely related information from the `CONTEXT` that might be helpful.
- If the `CONTEXT` is completely irrelevant, simply state that you cannot answer the query based on the provided documents.
- Keep your answer focused on the query. Do not ask follow-up questions.

CONTEXT:
{context}

USER QUERY:
{query} [/INST]"""
)

        def retrieve_and_format_context(inputs: Dict) -> str:
            query, cluster = inputs['query'], inputs['cluster']
            retriever = self.retrievers[cluster]
            retrieved_docs = retriever.invoke(query)
            inputs['retrieved_docs'] = retrieved_docs
            if not retrieved_docs:
                return "No relevant documents were found."
            return "\n\n".join(
                f"--- Document: {doc.metadata['source']} ---\n{doc.page_content}"
                for doc in retrieved_docs
            )
        
        self.chain = (
            RunnablePassthrough()
            | RunnablePassthrough.assign(context=RunnableLambda(retrieve_and_format_context))
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        logger.info("RAG service initialization complete.")

    def _identify_cluster(self, user_query: str) -> str:
        user_query_lower = user_query.lower()
        for cluster_name in self.settings.CLUSTERS.keys():
            if cluster_name in user_query_lower:
                return cluster_name
        return "unknown"


    def query(self, request: QueryRequest) -> QueryResponse:
        """Processes a user query, returning a clean answer and a separate list of sources."""
        if not self.chain or not self.retrievers:
            raise HTTPException(status_code=503, detail="Service Unavailable: RAG service is not initialized.")

        cluster = request.cluster if request.cluster in self.retrievers else self._identify_cluster(request.query)
        
        if cluster == "unknown" or cluster not in self.retrievers:
            cluster_list_str = ", ".join(self.retrievers.keys())
            return QueryResponse(
                answer=f"I couldn't determine which cluster you're asking about. Please specify one of: {cluster_list_str.capitalize()}.",
                cluster="unknown",
                sources=[]
            )
        
        logger.info(f"Processing query for cluster '{cluster}': '{request.query[:100]}...'")

        chain_input = {"query": request.query, "cluster": cluster, "retrieved_docs": []}
        try:
            llm_answer = self.chain.invoke(chain_input)
            retrieved_docs = chain_input['retrieved_docs']
        except Exception as e:
            logger.error(f"Error during RAG chain invocation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate a response from the model.")

        # Simplified source extraction logic. Assumes all retrieved docs are sources.
        # This is more robust than parsing the LLM output.
        source_objects = []
        if retrieved_docs:
            seen_titles = set()
            for doc in retrieved_docs:
                metadata = doc.metadata
                title = metadata.get('title', metadata.get('source', 'Unknown Source'))
                if title not in seen_titles:
                    source_objects.append(Source(
                        title=title,
                        url=metadata.get('url')
                    ))
                    seen_titles.add(title)
        
        return QueryResponse(
            answer=llm_answer.strip(),
            cluster=cluster,
            sources=source_objects
        )


# --- 4. FastAPI Application Setup ---
rag_service = RAGService(settings)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    rag_service.initialize()
    yield
    logger.info("Application shutdown.")

app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- 5. API Endpoints ---
@app.get("/", summary="Root endpoint")
async def root():
    return {"message": f"{settings.APP_TITLE} is running."}

@app.post("/query/", response_model=QueryResponse, summary="Query the knowledge base")
async def query_kb(request: QueryRequest):
    """
    Accepts a user query and returns a synthesized answer.
    The response body includes a separate list of source documents used to generate the answer.
    """
    return rag_service.query(request)