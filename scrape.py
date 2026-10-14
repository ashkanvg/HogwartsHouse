"""Step 1 - Gather pictures of the Harry Potter movie characters from the internet.

Where the pictures come from:
  * HP-API (https://hp-api.onrender.com): the list of characters, their house, the actor who
    played them in the movies, and one official picture.
  * The Harry Potter fandom wiki (https://harrypotter.fandom.com): many more pictures of each
    character (video-game screenshots, LEGO, logos and drawings are skipped).

Pictures are saved as data/raw/<House>/<Character_Name>/000.jpg, 001.jpg, ...
Characters that already have a folder are skipped, so you can stop and re-run any time.

Usage:
    python scrape.py                                        # every movie character, up to 50 pictures each
    python scrape.py --max-characters 8 --per-character 10  # quick test
"""
import argparse
import hashlib
import io
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from common import HOUSES, RAW_DIR

HP_API = "https://hp-api.onrender.com/api/characters"
WIKI_API = "https://harrypotter.fandom.com/api.php"

# File names on the wiki that are NOT photos of the character: games (WU = Wizards Unite,
# HM = Hogwarts Mystery, PAS = Puzzles & Spells, OOTPG = Order of the Phoenix game, ...), toys, logos, artwork, ...
SKIP_FILE = re.compile(
    r"lego|funko|vinyl|crest|logo|clearbg|icon|symbol|signature|concept|artwork|drawing|sketch|illustrat"
    r"|chocolate frog|card|cover|poster|book|map|statue|wand|cutscene|pottermore|jim kay|wallpaper|badge"
    r"|render|cartoon|animated|chibi|fan ?art|deviant|mystery|awakened|unite|legacy|puzzle"
    r"|\b(?:games?|wu|hm|ma|hl|pas|qc|mp|ppp|pop|ie|sig|psg|cosg|poag|gofg|ootpg|hbpg|dhg\d?"
    r"|gba|gbc|nds|ds|ps\d|pc|wii|xbox|b\d+c\d+m\d+)\b"
    r"|\.(?:gif|svg|webm|ogg|ogv)$",
    re.IGNORECASE,
)
# File names that are most likely movie stills - downloaded first (PSF = Philosopher's Stone film, ...)
FILM_FILE = re.compile(r"\b(?:psf|cosf|poaf|goff|ootpf|hbpf|dhf\d?|promo|film|movie)\b", re.IGNORECASE)
NOT_A_NAME = {"lord", "nearly", "moaning", "fat", "bloody"}  # "Fat Friar" shouldn't match "Fat Lady"

session = requests.Session()
session.headers["User-Agent"] = "HogwartsHouseSorter/1.0 (fun personal project; github.com/ashkanvg/HogwartsHouse)"
session.mount("https://", HTTPAdapter(pool_maxsize=16, max_retries=Retry(
    total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))


def get_characters():
    """Movie characters (they have an actor) who belong to one of the four houses."""
    characters, seen = [], set()
    for c in session.get(HP_API, timeout=60).json():
        if c["house"] not in HOUSES or not c["actor"]:
            continue
        if c["name"] in seen or c["actor"] in seen:  # the API lists a few characters twice
            continue
        seen.update([c["name"], c["actor"]])
        characters.append({"name": c["name"], "house": c["house"], "actor": c["actor"], "image": c["image"]})
    return characters


def words(text):
    return re.findall(r"[a-z]+", text.lower())


def make_name_matcher(characters):
    """Is a wiki file a picture of this character - and of nobody else?

    A file name mentions a character if it contains their full name, or a word of the name that no
    other character shares: "Draco" is enough for Draco Malfoy, but "Malfoy" alone could be Lucius.
    Files that mention a second character ("Harry and Cho kiss.jpg") are skipped: we can't tell
    whose face is whose.
    """
    counts = Counter(w for c in characters for w in set(words(c["name"])))
    keys = {c["name"]: (" ".join(words(c["name"])),
                        [w for w in words(c["name"]) if counts[w] == 1 and len(w) > 2 and w not in NOT_A_NAME])
            for c in characters}
    actors = {c["name"]: f" {' '.join(words(c['actor']))} " for c in characters}

    def mentions(title, name):
        full, unique = keys[name]
        return f" {full} " in title or any(f" {w} " in title for w in unique)

    def matches(file_title, name):
        # drop the character's own actor ("Katie Leung as Cho Chang" is not about Katie Bell)
        title = f" {' '.join(words(file_title))} ".replace(actors[name], " ")
        return mentions(title, name) and not any(mentions(title, other) for other in keys if other != name)

    return matches


def wiki_query(**params):
    return session.get(WIKI_API, params={"action": "query", "format": "json", **params}, timeout=30).json()


def wiki_file_titles(name):
    """All picture files on the character's wiki page plus those in 'Category:Images of <name>'."""
    data = wiki_query(titles=name, prop="images", imlimit="max", redirects=1)
    pages = list(data["query"]["pages"].values())
    titles = {img["title"] for page in pages for img in page.get("images", [])}
    page_title = pages[0].get("title", name)  # e.g. "Lord Voldemort" redirects to "Tom Riddle"
    params = {"list": "categorymembers", "cmtitle": f"Category:Images of {page_title}", "cmtype": "file", "cmlimit": "max"}
    while True:
        data = wiki_query(**params)
        titles.update(m["title"] for m in data.get("query", {}).get("categorymembers", []))
        if "continue" not in data:
            return titles
        params.update(data["continue"])


def search_wiki_page(character):
    """Some names lead to a disambiguation page ("Lily Potter"): search for the character's own page."""
    hits = wiki_query(list="search", srsearch=f"{character['name']} {character['actor']}", srlimit=5)
    for hit in hits.get("query", {}).get("search", []):
        if hit["title"] != character["actor"] and "disambiguation" not in hit["title"].lower():
            return hit["title"]
    return None


def wiki_image_urls(titles, width=640):
    """Download links for wiki files (as thumbnails at most `width` pixels wide), in the same order."""
    urls = []
    for i in range(0, len(titles), 50):  # the API answers 50 files per request
        batch = titles[i:i + 50]
        data = wiki_query(titles="|".join(batch), prop="imageinfo", iiprop="url|mime", iiurlwidth=width)
        info = {p["title"]: p["imageinfo"][0] for p in data["query"]["pages"].values() if p.get("imageinfo")}
        for title in batch:
            if title in info and info[title].get("mime") in ("image/jpeg", "image/png", "image/webp"):
                urls.append(info[title].get("thumburl") or info[title]["url"])
    return urls


def download_image(url):
    """Download one picture. Returns a PIL image, or None if it is broken or too small."""
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception:
        return None
    if min(img.size) < 120:
        return None
    img.thumbnail((640, 640))
    return img


def fingerprint(img):
    """Tiny 8x8 'average hash': the same for resized copies of one picture, so we skip duplicates."""
    pixels = list(img.convert("L").resize((8, 8)).tobytes())
    average = sum(pixels) / len(pixels)
    return tuple(p > average for p in pixels)


def scrape_character(character, per_character, matches):
    folder = RAW_DIR / character["house"] / re.sub(r"[^A-Za-z0-9]+", "_", character["name"]).strip("_")
    if pictures := list(folder.glob("*.jpg")):
        return len(pictures), "already downloaded"

    def useful(file_titles):
        return {t for t in file_titles if matches(t, character["name"]) and not SKIP_FILE.search(t)}

    titles = useful(wiki_file_titles(character["name"]))
    if len(titles) < 5 and (page := search_wiki_page(character)):
        titles |= useful(wiki_file_titles(page))
    # movie stills first, then the rest in a fixed shuffled order
    titles = sorted(titles, key=lambda t: (FILM_FILE.search(t) is None, hashlib.md5(t.encode()).hexdigest()))
    urls = ([character["image"]] if character["image"] else []) + wiki_image_urls(titles[:per_character * 2])

    folder.mkdir(parents=True, exist_ok=True)
    sources, seen = {}, set()
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        for url, img in zip(urls, pool.map(download_image, urls)):
            if img is None or (key := fingerprint(img)) in seen:
                continue
            seen.add(key)
            filename = f"{len(sources):03d}.jpg"
            img.save(folder / filename, quality=90)
            sources[filename] = url
            if len(sources) >= per_character:
                break
    finally:
        pool.shutdown(cancel_futures=True)
    (folder / "sources.json").write_text(json.dumps(sources, indent=1))  # where each picture came from
    return len(sources), f"{len(titles)} matching files on the wiki"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-character", type=int, default=50, help="max pictures per character (default 50)")
    parser.add_argument("--max-characters", type=int, default=0, help="only the first N characters (0 = all)")
    args = parser.parse_args()

    print("Asking HP-API for the characters...")
    all_characters = get_characters()
    characters = all_characters[:args.max_characters] if args.max_characters else all_characters
    print(f"Found {len(all_characters)} movie characters with a house.\n")

    matches = make_name_matcher(all_characters)
    totals = Counter()
    for i, character in enumerate(characters, 1):
        try:
            count, note = scrape_character(character, args.per_character, matches)
        except (requests.RequestException, KeyError, ValueError) as error:  # one bad answer shouldn't stop the run
            count, note = 0, f"error: {error!r}"
        character["pictures"] = count
        totals[character["house"]] += count
        print(f"[{i:3d}/{len(characters)}] {character['house']:<10} {character['name']:<28} {count:3d} pictures  ({note})")
        time.sleep(0.3)  # be gentle with the wiki

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "characters.json").write_text(json.dumps(characters, indent=2))
    print("\nPictures per house:", ", ".join(f"{h} {totals[h]}" for h in HOUSES))
    print(f"Saved in {RAW_DIR}. Next step: python prepare.py")


if __name__ == "__main__":
    main()
