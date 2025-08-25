import os
import shutil
import subprocess
import yaml
import csv
import re
import sys
import logging
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from pprint import pprint
from typing import Iterator, Dict, Any, Optional


# --- CONFIGURATION ---
SOURCES = [
    {
        "repo_url": "git@github.com:stanford-rc/farmshare-docs.git",
        "repo_name": "farmshare",
    },
    {
        "repo_url": "git@github.com:stanford-rc/docs.elm.stanford.edu.git",
        "repo_name": "elm",
    },
     {
        "repo_url": "git@github.com:stanford-rc/docs.oak.stanford.edu.git",
        "repo_name": "oak",
    },
    {
        "repo_url": "git@github.com:stanford-rc/www.sherlock.stanford.edu.git",
        "repo_name": "sherlock",
        "scraper_targets": [
            {
                "url": "https://www.sherlock.stanford.edu/docs/tech/facts/",
                "file": "sherlock/facts.md",
                "title": "Sherlock Facts"
            },
            {
                "url": "https://www.sherlock.stanford.edu/docs/tech/",
                "file": "sherlock/tech.md",
                "title": "Sherlock Technical Documentation"
            },
            {
                "url": "https://www.sherlock.stanford.edu/docs/software/list/",
                "file": "sherlock/list.md",
                "title": "Sherlock Software List"
            },
            {
                "url": "https://www.sherlock.stanford.edu/docs/",
                "file": "sherlock/index.md",
                "title": "Welcome to Sherlock"
            },
            {
                "url": "https://www.sherlock.stanford.edu",
                "file": "sherlock/home.md",
                "title": "Sherlock"
            },
        ]
    },
]
# Configure logging
LOG_FILE = 'magicFile.log'
logging.basicConfig(level=logging.INFO,
                    filename=LOG_FILE,
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
# --- General-Purpose Tolerant YAML Loader ---
def ignore_and_warn_on_unknown_tags(loader, tag_prefix, node):
    logging.warning(f"Ignoring unknown YAML tag '{node.tag}'")
    return None
class TolerantSafeLoader(yaml.SafeLoader):
    pass
TolerantSafeLoader.add_multi_constructor('!', ignore_and_warn_on_unknown_tags)
TolerantSafeLoader.add_multi_constructor('tag:yaml.org,2002:python', ignore_and_warn_on_unknown_tags)


# --- CORE FUNCTIONS ---

# --- NEW FUNCTION ---
def resolve_markdown_references(content: str) -> str:
    """
    Resolves Markdown reference-style links.
    e.g., turns `[text][ref]` into `[text](url)` and removes the `[ref]: url` definitions.
    """
    # 1. Find all reference definitions, e.g., `[ref_name]: http://...`
    # This pattern also handles optional titles like `[ref]: url "title"`
    definition_pattern = re.compile(r"^\s*\[([^\]]+)\]:\s*([^\s]+)(?:\s+[\"'(].*[\"')])?\s*$", re.MULTILINE)
    definitions = {ref.lower(): url for ref, url in definition_pattern.findall(content)}
    
    if not definitions:
        return content # No references to resolve

    logging.info(f"    - Found {len(definitions)} reference-style link definitions.")

    # 2. Replace all reference usages, e.g., `[My Link][ref_name]`
    usage_pattern = re.compile(r'(\[([^\]]+)\])\[([^\]]+)\]')

    def replacer(match):
        full_match, text_part, text_content, ref_name = match.groups(), match.group(1), match.group(2), match.group(3)
        
        # Handle implicit refs like `[ref_name]` where text is the ref
        if not ref_name:
             ref_name = text_content
        
        url = definitions.get(ref_name.lower().strip())
        if url:
            logging.info(f"    - Resolving ref '{ref_name}' to inline link.")
            return f"[{text_content}]({url})"
        else:
            # If a reference is not found, leave it as is to avoid breaking content
            return match.group(0)

    # 3. Handle shorthand links where the text is the reference, e.g. [myref]
    shorthand_pattern = re.compile(r'\[([^\]]+)\](?!\[|:|\()') # Negative lookaheads to avoid matching other link types

    def shorthand_replacer(match):
        ref_name = match.group(1)
        url = definitions.get(ref_name.lower().strip())
        if url:
            logging.info(f"    - Resolving shorthand ref '{ref_name}' to inline link.")
            return f"[{ref_name}]({url})"
        return match.group(0) # Not a valid shorthand ref, leave it

    # Apply replacements
    content = usage_pattern.sub(replacer, content)
    content = shorthand_pattern.sub(shorthand_replacer, content)

    # 4. Remove the original definition lines from the content
    content = definition_pattern.sub('', content).strip()
    
    return content


def expand_markdown_links(content: str, base_url: str) -> str:
    """Finds all relative Markdown links and expands them to absolute URLs."""
    link_pattern = re.compile(r'\[([^\]]+)\]\((?!(#|mailto:|tel:))([^\)]+)\)')
    def replacer(match):
        text, _, url = match.groups()
        parsed_url = urlparse(url.strip())
        if not parsed_url.scheme and not parsed_url.netloc:
            absolute_url = urljoin(base_url, url)
            logging.info(f"    - Expanded MD link: '{url}' -> '{absolute_url}'")
            return f'[{text}]({absolute_url})'
        return match.group(0)
    return link_pattern.sub(replacer, content)

# --- REFACTORED FUNCTION ---
def process_and_write_markdown_file(source_path: Path, dest_path: Path, meta: dict):
    """
    Reads a source Markdown file, resolves references, expands relative links,
    adds YAML front matter, and writes the result.
    """
    try:
        content = source_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        logging.warning(f"Source file not found: {source_path}. Skipping processing.")
        return

    front_matter_pattern = re.compile(r'^---\s*\n(.*?\n)---\s*\n', re.DOTALL)
    match = front_matter_pattern.match(content)

    existing_metadata, main_content = {}, content
    if match:
        try:
            existing_metadata = yaml.load(match.group(1), Loader=TolerantSafeLoader) or {}
        except yaml.YAMLError as e:
            logging.warning(f"Could not parse existing front matter in {source_path}: {e}")
        main_content = content[match.end():]

    # --- NEW: Resolve reference-style links FIRST ---
    main_content = resolve_markdown_references(main_content)

    # --- THEN, expand any relative links (including those from the resolved refs) ---
    page_base_url = meta.get('url')
    if page_base_url:
        main_content = expand_markdown_links(main_content, page_base_url)
    else:
        logging.warning(f"No 'url' in metadata for {source_path}, cannot expand relative links.")

    existing_metadata.update(meta)
    new_yaml_front_matter = yaml.dump(existing_metadata, default_flow_style=False, sort_keys=False)
    new_content = f"---\n{new_yaml_front_matter}---\n\n{main_content.lstrip()}"
    dest_path.write_text(new_content, encoding='utf-8')


# --- Other functions (clone_repo, parse_nav_generator, scrape_url_to_file, etc.) ---
# --- are unchanged and omitted for brevity. You can copy them from the previous version. ---
def clone_repo(repo_url: str, local_path: Path):
    """Clones a GitHub repository, ensuring a fresh start."""
    logging.info(f"Cloning repository: {repo_url} into {local_path}")
    if local_path.exists():
        logging.info(f"Removing existing directory: {local_path}")
        shutil.rmtree(local_path)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(local_path)],
            check=True, capture_output=True, text=True
        )
        logging.info("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error cloning repository: {e.stderr}")
        raise

def generate_url_from_path(relative_path: str, base_url: str) -> str:
    """Generates a clean, 'pretty' URL from a file path."""
    if not base_url.endswith('/'):
        base_url += '/'
    url_path = Path(relative_path).as_posix() # Use posix paths for URLs
    if url_path.endswith('index.md'):
        url_path = str(Path(url_path).parent) + '/'
    else:
        url_path = str(Path(url_path).with_suffix('')) + '/'
    if url_path in ('./', '/'):
        url_path = ''
    return urljoin(base_url, url_path)

def parse_nav_generator(node: Any, docs_dir: Path, current_category: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Recursively parses a navigation node and yields a dictionary for each document."""
    if isinstance(node, list):
        for item in node:
            yield from parse_nav_generator(item, docs_dir, current_category)
    elif isinstance(node, dict):
        for title, path_or_list in node.items():
            if isinstance(path_or_list, list):
                yield from parse_nav_generator(path_or_list, docs_dir, current_category=title)
            elif isinstance(path_or_list, str):
                yield {
                    "title": title, "relative_path": path_or_list,
                    "source_path": docs_dir / path_or_list, "file_name": Path(path_or_list).name
                }
    elif isinstance(node, str) and current_category:
        yield {
            "title": current_category, "relative_path": node,
            "source_path": docs_dir / node, "file_name": Path(node).name
        }

def scrape_url_to_file(url: str, output_path: Path, title: str):
    """Scrapes HTML, expands relative links, converts to Markdown, and saves."""
    logging.info(f"-> Processing URL: {url}")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content_div = soup.find('div', role='main') or soup.find('article') or soup.find('section', id='info')
        if not main_content_div:
            logging.error(f"Could not find a suitable main content container on {url}")
            return

        for a_tag in main_content_div.find_all('a', href=True):
            href = a_tag.get('href')
            parsed_href = urlparse(href)
            if not parsed_href.scheme and not parsed_href.netloc:
                absolute_url = urljoin(url, href)
                a_tag['href'] = absolute_url
                logging.info(f"    - Expanded HTML link: '{href}' -> '{absolute_url}'")

        markdown_content = md(str(main_content_div), heading_style="ATX")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown_content, encoding='utf-8')
        metadata_to_add = {'title': title, 'url': url}
        process_and_write_markdown_file(output_path, output_path, metadata_to_add)
        logging.info(f"✅ Success for: {output_path}")

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ ERROR fetching {url}: {e}")
    except Exception as e:
        logging.error(f"❌ An unexpected ERROR occurred for {url}: {e}", exc_info=True)


def handle_duplicate_filename(filename: str, used_filenames: set) -> str:
    """Checks for and resolves duplicate filenames by appending a counter."""
    if filename not in used_filenames:
        used_filenames.add(filename)
        return filename
    p = Path(filename)
    base, ext = p.stem, p.suffix
    counter = 1
    while True:
        new_filename = f"{base}-{counter}{ext}"
        if new_filename not in used_filenames:
            used_filenames.add(new_filename)
            return new_filename
        counter += 1

def variable_clean_up(dest_path: Path):
    """Wrapper for the variable cleanup process."""
    try:
        process_markdown_file(dest_path)
    except Exception as e:
        logging.error(f"Failed to run variable cleanup on {dest_path}: {e}")

def cleanup_directory(dir_path: Path):
    """Removes a directory if it exists."""
    if dir_path.exists():
        logging.info(f"Removing temporary directory: {dir_path}")
        shutil.rmtree(dir_path)

def process_repository(config: dict):
    """Main processing logic for a single repository."""
    repo_url, repo_name = config["repo_url"], config["repo_name"]
    scraper_targets = config.get("scraper_targets", [])
    print(f"\n{'='*20} Processing Repository: {repo_name} {'='*20}")
    logging.info(f"Starting processing for repository: {repo_url}")

    local_repo_path = Path(f"temp_repo_{repo_name}")
    flat_output_dir = Path(repo_name)
    output_csv_file = Path(f"{repo_name}_url_map.csv")

    try:
        clone_repo(repo_url, local_repo_path)
    except Exception:
        print(f"FATAL: Halting processing for {repo_name} due to repository clone failure.")
        logging.critical(f"Clone failure for {repo_name}. Aborting its processing.")
        cleanup_directory(local_repo_path)
        return

    config_file = local_repo_path / "mkdocs.yml"
    if not config_file.exists():
        print(f"Error: 'mkdocs.yml' not found in {repo_name}. Skipping mkdocs processing.")
        logging.error(f"'mkdocs.yml' not found for {repo_name}.")
    else:
        process_mkdocs_repo(config_file, local_repo_path, flat_output_dir, output_csv_file)

    if scraper_targets:
        print("\n--- Starting Scraper ---")
        for i, target in enumerate(scraper_targets, 1):
            print(f"\n[{i}/{len(scraper_targets)}] Scraping '{target.get('title')}'")
            url, file_path_str, title = target.get("url"), target.get("file"), target.get("title")
            if not all([url, file_path_str, title]):
                logging.warning(f"Skipping invalid scraper target entry: {target}")
                continue
            scrape_output_path = flat_output_dir / Path(file_path_str).name
            scrape_url_to_file(url=url, output_path=scrape_output_path, title=title)
    else:
        print("No scraper targets defined for this repository.")

    cleanup_directory(local_repo_path)
    print(f"--- Finished processing for {repo_name} ---")


def process_mkdocs_repo(config_file: Path, repo_path: Path, flat_dir_path: Path, output_csv_file: Path):
    """Handles processing for mkdocs repos, including link expansion."""
    print(f"Loading configuration from: {config_file}")
    try:
        config = yaml.load(config_file.read_text(encoding='utf-8'), Loader=TolerantSafeLoader)
    except yaml.YAMLError as e:
        print(f"FATAL: YAML syntax error in {config_file}. Details: {e}")
        logging.critical(f"YAML syntax error in {config_file}: {e}")
        return

    base_site_url = config.get('site_url')
    docs_dir_name = config.get('docs_dir', 'docs')
    nav = config.get('nav')
    if not all([base_site_url, nav]):
        print("Error: 'site_url' or 'nav' not found in mkdocs.yml. Skipping doc generation.")
        logging.error("'site_url' or 'nav' missing in mkdocs.yml.")
        return

    docs_dir = repo_path / docs_dir_name
    print(f"Successfully parsed config. Site URL: {base_site_url}, Docs dir: {docs_dir}")

    flat_dir_path.mkdir(exist_ok=True)
    used_filenames = set()
    processed_count = 0

    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['url', 'file_name', 'title']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()

        print("\nStarting processing of documents from navigation...")
        for doc_info in parse_nav_generator(nav, docs_dir):
            doc_info['url'] = generate_url_from_path(doc_info['relative_path'], base_site_url)
            doc_info['file_name'] = handle_duplicate_filename(doc_info['file_name'], used_filenames)
            csv_writer.writerow({k: doc_info[k] for k in fieldnames})

            destination_path = flat_dir_path / doc_info['file_name']
            print(f"  - Processing: {doc_info['relative_path']:<40} -> {destination_path}")

            metadata_to_add = {'title': doc_info['title'], 'url': doc_info['url']}
            process_and_write_markdown_file(doc_info['source_path'], destination_path, metadata_to_add)
            variable_clean_up(destination_path)
            processed_count += 1

    print(f"\nSuccessfully processed {processed_count} documents from mkdocs.yml.")
    print(f"URL map written to {output_csv_file}")
    print(f"Markdown documents saved in {flat_dir_path}")

def main():
    """Main execution function."""
    print("--- Document Processing Script Started ---")
    logging.info("Script started.")
    if not SOURCES:
        print("No sources configured in the 'SOURCES' list. Exiting.")
        logging.warning("SOURCES list is empty. Nothing to do.")
        return
    for source_config in SOURCES:
        try:
            process_repository(source_config)
        except Exception as e:
            repo_name = source_config.get("repo_name", "Unknown")
            print(f"An unexpected error occurred while processing {repo_name}. See log for details.")
            logging.error(f"CRITICAL FAILURE during processing of {repo_name}: {e}", exc_info=True)
    print("\n--- All configured jobs have been processed. ---")
    logging.info("Script finished successfully.")


if __name__ == "__main__":
    main()