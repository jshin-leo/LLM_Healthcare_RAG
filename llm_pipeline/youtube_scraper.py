# ---------------------------------------------------------------------
#   Project: LLM for HealthCare
#
#   - Title: llm_pipeline/youtube_scraper.py
#   - Author: Jihoon Shin
#   - Date: June 5th, 2025
#   - What: Crawl websites to extract YouTube links from all subpages
#   - How: 
#       1. Provide starting URLs at TARGETS. (e.g., resource center)
#       2. This script crawls internal links and gathers YouTube URLs.
#       3. Saves results to youtube_links.txt
# ---------------------------------------------------------------------

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def is_allowed_path(link, allowed_prefix):
    parsed = urlparse(link)
    return parsed.path.startswith(allowed_prefix)

def clean_url(url):
    return url.split('#')[0].strip('/')

def extract_internal_links(soup, base_url, allowed_prefix):
    links = set()
    for tag in soup.find_all("a", href=True):
        full_url = urljoin(base_url, tag["href"])
        if full_url.startswith("http") and is_allowed_path(full_url, allowed_prefix):
            links.add(clean_url(full_url))
    return links

def extract_youtube_links(soup):
    links = set()
    for a in soup.find_all("a", href=True):
        if "youtube.com/watch" in a["href"] or "youtu.be/" in a["href"]:
            links.add(a["href"])
    for iframe in soup.find_all("iframe", src=True):
        if "youtube.com/embed/" in iframe["src"] or "youtube.com/watch" in iframe["src"]:
            links.add(iframe["src"])
    return links

def crawl_youtube_links_and_save(targets, output_file, max_pages=10000, verbose=True):
    """
    Crawl provided websites and directly append discovered YouTube links to a file.

    Args:
        targets (list): List of (start_url, allowed_path_prefix)
        output_file (str): Path to save YouTube links incrementally
        max_pages (int): Max number of pages to crawl
        verbose (bool): Print crawl info
    """
    visited_urls = set()
    discovered_links = set()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9"
    }

    # Make sure file is clean or created
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")

    for start_url, allowed_prefix in targets:
        queue = [start_url]
        page_count = 0

        if verbose:
            print(f"\nStarting crawl from: {start_url} (Allowed path: {allowed_prefix})")

        while queue and page_count < max_pages:
            current_url = queue.pop(0)
            if current_url in visited_urls:
                continue

            visited_urls.add(current_url)
            page_count += 1

            try:
                response = requests.get(current_url, timeout=10, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                found_links = extract_youtube_links(soup)
                new_links = found_links - discovered_links
                if new_links:
                    discovered_links.update(new_links)
                    with open(output_file, "a", encoding="utf-8") as f:
                        for link in sorted(new_links):
                            f.write(link + "\n")
                    if verbose:
                        print(f"[{page_count}] Added {len(new_links)} new link(s) from {current_url}")

                internal_links = extract_internal_links(soup, current_url, allowed_prefix)
                queue.extend(link for link in internal_links if link not in visited_urls)

            except Exception as e:
                if verbose:
                    print(f"[ERROR] Skipping {current_url}: {e}")

# previous code for crawl.
def crawl_youtube_links(targets, max_pages=10000, verbose=True):
    """
    Crawl provided websites to extract YouTube links.

    Args:
        targets (list): List of (start_url, allowed_path_prefix)
        max_pages (int): Maximum number of pages to crawl
        verbose (bool): Whether to print crawl progress

    Returns:
        set: Unique YouTube links found
    """
    visited_urls = set()
    youtube_links = set()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9"
    }

    for start_url, allowed_prefix in targets:
        queue = [start_url]
        page_count = 0

        if verbose:
            print(f"\nStarting crawl from: {start_url} (Allowed path: {allowed_prefix})")

        while queue and page_count < max_pages:
            current_url = queue.pop(0)
            if current_url in visited_urls:
                continue

            visited_urls.add(current_url)
            page_count += 1

            try:
                response = requests.get(current_url, timeout=10, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract YouTube links
                found_links = extract_youtube_links(soup)
                if found_links:
                    youtube_links.update(found_links)
                    if verbose:
                        print(f"[{page_count}] Found {len(found_links)} YouTube link(s) at: {current_url}")

                # Queue internal links
                new_links = extract_internal_links(soup, current_url, allowed_prefix)
                for link in new_links:
                    if link not in visited_urls:
                        queue.append(link)

            except Exception as e:
                if verbose:
                    print(f"[ERROR] Skipping {current_url}: {e}")

    return youtube_links

def save_links_to_file(links, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for link in sorted(links):
            f.write(link + "\n")
    print(f"Saved {len(links)} YouTube links to '{output_file}'")