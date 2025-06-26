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
import subprocess
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import llm_pipeline.utils as utils


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


def save_links_to_file(links, output_file, overwrite=True):
    mode = "w" if overwrite else "a"
    with open(output_file, mode, encoding="utf-8") as f:
        for link in sorted(links):
            f.write(link + "\n")
    print(f"Saved {len(links)} YouTube links to '{output_file}'")


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
                    save_links_to_file(new_links, output_file, overwrite=False)
                    
                    if verbose:
                        print(f"[{page_count}] Added {len(new_links)} new link(s) from {current_url}")

                internal_links = extract_internal_links(soup, current_url, allowed_prefix)
                queue.extend(link for link in internal_links if link not in visited_urls)

            except Exception as e:
                if verbose:
                    print(f"[ERROR] Skipping {current_url}: {e}")

# To extract youtube links from Playlists. 
def extract_playlist_links(input_file, output_file):
    new_links = set()  # Use a set to prevent duplicates

    with open(input_file, "r", encoding="utf-8") as f:
        playlist_urls = [line.strip() for line in f if line.strip()]

    # Make sure file is clean or created
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")

    for playlist_url in playlist_urls:
        print(f"Processing playlist: {playlist_url}")
        command = [
            "yt-dlp",
            "--flat-playlist",
            "--dump-json",
            playlist_url
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            for line in result.stdout.strip().split('\n'):
                try:
                    video_info = json.loads(line)
                    video_id = video_info.get("id")
                    if video_id:
                        new_links.add(f"https://www.youtube.com/watch?v={video_id}")

                except Exception as e:
                    print(f"[ERROR] Could not parse line: {e}")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] yt-dlp failed on {playlist_url}: {e}")

    # Save all collected unique links to the output file
    save_links_to_file(new_links, output_file, overwrite=True)
    
    print(f"\nExtracted {len(new_links)} video links to '{output_file}'")
    
