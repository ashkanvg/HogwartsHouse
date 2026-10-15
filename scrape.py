"""Step 1 - Gather pictures of the Harry Potter movie characters from the internet.

Where the pictures come from:
  * HP-API (https://hp-api.onrender.com): the list of characters, their house, the actor who
    played them in the movies, and one official picture.
  * The Harry Potter fandom wiki (https://harrypotter.fandom.com): many more pictures of each
    character (video-game screenshots, LEGO, logos and drawings are skipped).
  * With --commons: real-life photos of the actors from Wikimedia Commons
    (https://commons.wikimedia.org), only with free licences (Creative Commons or public domain).
    Wikidata tells us which Commons category belongs to which actor.

Pictures are saved as data/raw/<House>/<Character_Name>/000.jpg, 001.jpg, ... and the actor
photos as commons_000.jpg, ... Each folder's sources.json says where every picture came from
(and, for Commons photos, who took it and under which licence).
Characters that already have pictures are skipped, so you can stop and re-run any time.

Usage:
    python scrape.py                                        # every movie character, up to 50 pictures each
    python scrape.py --commons                              # ...plus up to 25 photos of each actor
    python scrape.py --max-characters 8 --per-character 10  # quick test
"""
import argparse
import hashlib
import io
import json
import re
import time
import urllib.parse
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


def character_folder(character):
    return RAW_DIR / character["house"] / re.sub(r"[^A-Za-z0-9]+", "_", character["name"]).strip("_")


def scrape_character(character, per_character, matches):
    folder = character_folder(character)
    if pictures := [p for p in folder.glob("*.jpg") if not p.name.startswith(("commons_", "roboflow_"))]:
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
    sources_file = folder / "sources.json"
    sources = json.loads(sources_file.read_text()) if sources_file.exists() else {}  # keeps the actor photos' credits
    count, seen = 0, set()
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        for url, img in zip(urls, pool.map(download_image, urls)):
            if img is None or (key := fingerprint(img)) in seen:
                continue
            seen.add(key)
            filename = f"{count:03d}.jpg"
            img.save(folder / filename, quality=90)
            sources[filename] = url
            count += 1
            if count >= per_character:
                break
    finally:
        pool.shutdown(cancel_futures=True)
        sources_file.write_text(json.dumps(sources, indent=1))  # where each picture came from
    return count, f"{len(titles)} matching files on the wiki"


# ---------------------------------------------------------------- real-life photos of the actors (--commons)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
FILM_SERIES = ["Q216930", "Q27922181"]  # Wikidata items: the Harry Potter films, the Fantastic Beasts films
BRITISH_ACTOR = re.compile(r"\b(english|british|scottish|welsh|irish)\b.*\b(actor|actress)\b", re.IGNORECASE)
NOT_JUST_THEM = re.compile(  # wax figures, signatures, art, and photos of several people
    r"wax|tussaud|signature|autograph|drawing|painting|sculpture|statue|cosplay|poster|mural|graffiti"
    r"|caricature|\bart\b| & | and |\bwith\b", re.IGNORECASE)
# CC BY and CC BY-SA (any version), CC0 and public domain - not the "non-commercial" or "no derivatives" kinds
FREE_LICENCE = re.compile(r"^(?!.*\bn[cd]\b)(?:cc0|cc[ -]by(?:[ -]sa)?\b|public domain|pd\b)", re.IGNORECASE)
PHOTO_WIDTH = 500  # one of Wikimedia's standard thumbnail sizes, cheap for it to make
BUSY_PAUSE = 60    # seconds to wait when Wikimedia says it's too busy

# Photo downloads get no automatic retries: if Wikimedia says "too busy", we pause and move on
commons_session = requests.Session()
commons_session.headers["User-Agent"] = session.headers["User-Agent"]


def polite_get(url, **params):
    """Wikimedia asks for one request at a time, so wait a moment before each one."""
    time.sleep(1)
    response = session.get(url, params=params, timeout=120)
    response.raise_for_status()
    return response.json()


def plain_name(name):
    """'Dame Maggie Smith' -> 'maggie smith', so names written differently still match."""
    return " ".join(re.findall(r"[a-z]+", re.sub(r"^(dame|sir)\s+", "", name.strip().lower())))


def commons_category(p373=None, commons_page=None):
    """A Commons category name from Wikidata: its 'Commons category' value, or a link to a category page."""
    if p373:
        return p373
    if commons_page and "/wiki/Category:" in commons_page:
        return urllib.parse.unquote(commons_page.split("/wiki/Category:", 1)[1]).replace("_", " ")
    return None


def film_cast_categories():
    """{plain actor name: Commons category} for everyone in the films' cast lists on Wikidata."""
    series = " ".join(f"wd:{q}" for q in FILM_SERIES)
    query = f"""SELECT DISTINCT ?name ?category ?page WHERE {{
      VALUES ?series {{ {series} }} ?film wdt:P179 ?series ; wdt:P161 ?actor .
      {{ ?actor rdfs:label ?name }} UNION {{ ?actor skos:altLabel ?name }} FILTER(LANG(?name) = "en")
      OPTIONAL {{ ?actor wdt:P373 ?category }}
      OPTIONAL {{ ?page schema:about ?actor ; schema:isPartOf <https://commons.wikimedia.org/> }}
    }}"""
    rows = polite_get(WIKIDATA_SPARQL, query=query, format="json")["results"]["bindings"]
    cast = {}
    for row in rows:
        category = commons_category(row.get("category", {}).get("value"), row.get("page", {}).get("value"))
        if category:
            cast[plain_name(row["name"]["value"])] = category
    return cast


def entity_category(entity):
    """The Commons category of a Wikidata item (from wbgetentities), or None."""
    claims = entity.get("claims", {}).get("P373", [])
    if claims:
        return claims[0]["mainsnak"]["datavalue"]["value"]
    page = entity.get("sitelinks", {}).get("commonswiki", {}).get("title", "")
    return page.removeprefix("Category:") if page.startswith("Category:") else None


def actor_category(actor, cast):
    """The Commons category with photos of this actor: from the films' cast list, or else from a
    British or Irish actor/actress with that name on Wikidata (some cast lists there are incomplete)."""
    if plain_name(actor) in cast:
        return cast[plain_name(actor)]
    name = re.sub(r"^(Dame|Sir)\s+", "", actor)
    results = polite_get(WIKIDATA_API, action="wbsearchentities", search=name, language="en", type="item",
                         limit=7, format="json").get("search", [])
    ids = [r["id"] for r in results if BRITISH_ACTOR.search(r.get("description", ""))][:3]
    if not ids:
        return None
    entities = polite_get(WIKIDATA_API, action="wbgetentities", ids="|".join(ids), props="claims|sitelinks",
                          format="json")["entities"]
    return next((c for i in ids if (c := entity_category(entities.get(i, {})))), None)


def usable_photos(pages, other_actors):
    """Freely licensed JPEG/PNG photos of just this person, best search matches first.
    No wax figures, signatures or art, and no photos naming several people."""
    photos = []
    for page in sorted(pages, key=lambda p: p.get("index", 0)):
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        licence = meta.get("LicenseShortName", {}).get("value", "")
        title = f" {plain_name(page['title'])} "
        if (info.get("mime") not in ("image/jpeg", "image/png") or NOT_JUST_THEM.search(page["title"])
                or not FREE_LICENCE.search(licence) or any(f" {other} " in title for other in other_actors)):
            continue
        photos.append({"url": info.get("thumburl") or info["url"], "page": info.get("descriptionurl", ""),
                       "licence": licence,
                       "author": re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()})
    return photos


def commons_photos(category, limit, other_actors):
    """Photos in this Commons category and the categories inside it (like 'Daniel Radcliffe in 2017')."""
    photos, offset = [], 0
    while len(photos) < limit:
        data = polite_get(COMMONS_API, action="query", format="json", generator="search", gsrnamespace=6,
                          gsrsearch=f'deepcat:"{category}" filetype:bitmap', gsrlimit=50, gsroffset=offset,
                          prop="imageinfo", iiprop="url|mime|extmetadata", iiurlwidth=PHOTO_WIDTH,
                          iiextmetadatafilter="LicenseShortName|Artist")
        photos += usable_photos(data.get("query", {}).get("pages", {}).values(), other_actors)
        offset = data.get("continue", {}).get("gsroffset")  # the next page of search results, if any
        if offset is None:
            break
    return photos[:limit]


def download_commons_photo(url):
    """Download one Commons photo. Returns (PIL image or None, whether Wikimedia said it's too busy)."""
    try:
        response = commons_session.get(url, timeout=60)
    except requests.RequestException:
        return None, False
    if response.status_code in (429, 503):
        return None, True
    try:
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception:
        return None, False
    return (img if min(img.size) >= 120 else None), False


def scrape_actor_photos(character, category, limit, other_actors):
    """Add real-life photos of the character's actor to the character's folder.

    Running it again tops up actors that got fewer photos (for example when Wikimedia was busy),
    without downloading the same photo twice. It also writes down the photographer and licence
    again for photos whose credit is missing from sources.json.
    """
    folder = character_folder(character)
    sources_file = folder / "sources.json"
    sources = json.loads(sources_file.read_text()) if sources_file.exists() else {}
    saved = len(list(folder.glob("commons_*.jpg")))
    uncredited = {fingerprint(Image.open(p).convert("RGB")): p.name
                  for p in folder.glob("commons_*.jpg") if p.name not in sources}
    if saved >= limit and not uncredited:
        return saved, "already downloaded"
    folder.mkdir(parents=True, exist_ok=True)
    have = {s["url"] for s in sources.values() if isinstance(s, dict)}  # Commons photos we already have
    photos = [p for p in commons_photos(category, limit * 2, other_actors) if p["page"] not in have]
    seen = {fingerprint(Image.open(p).convert("RGB")) for p in folder.glob("*.jpg")}  # skip copies of our pictures
    busy = 0
    for photo in photos:
        if saved >= limit and not uncredited:
            break
        time.sleep(2)  # one download at a time, gently: Wikimedia turns away clients that hurry
        img, too_busy = download_commons_photo(photo["url"])
        if too_busy:
            busy += 1
            if busy == 3:  # leave the rest for a later run, which tops up
                break
            time.sleep(BUSY_PAUSE)
            continue
        if img is None:
            continue
        credit = {"url": photo["page"], "author": photo["author"], "licence": photo["licence"]}
        if (key := fingerprint(img)) in uncredited:  # a photo we have, but its credit got lost
            sources[uncredited.pop(key)] = credit
        elif key in seen or saved >= limit:
            continue
        else:
            seen.add(key)
            name = f"commons_{saved:03d}.jpg"
            img.save(folder / name, quality=90)
            sources[name] = credit
            saved += 1
        sources_file.write_text(json.dumps(sources, indent=1))  # straight away: a stopped run keeps every credit
    return saved, f"Commons Category:{category}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-character", type=int, default=50, help="max pictures per character (default 50)")
    parser.add_argument("--max-characters", type=int, default=0, help="only the first N characters (0 = all)")
    parser.add_argument("--commons", action="store_true",
                        help="also add free-licensed real-life photos of the actors from Wikimedia Commons")
    parser.add_argument("--per-actor", type=int, default=25, help="max Commons photos per actor (default 25)")
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

    if args.commons:
        print("\nAdding free-licensed photos of the actors from Wikimedia Commons...")
        cast = film_cast_categories()
        actors = {plain_name(c["actor"]) for c in all_characters}
        for i, character in enumerate(characters, 1):
            try:
                category = actor_category(character["actor"], cast)
                if category is None:
                    count, note = 0, "no photos of the actor on Commons"
                else:
                    others = actors - {plain_name(character["actor"])}
                    count, note = scrape_actor_photos(character, category, args.per_actor, others)
            except (requests.RequestException, KeyError, ValueError) as error:
                count, note = 0, f"error: {error!r}"
            character["actor_photos"] = count
            totals[character["house"]] += count
            print(f"[{i:3d}/{len(characters)}] {character['house']:<10} {character['actor']:<28} {count:3d} photos  ({note})")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "characters.json").write_text(json.dumps(characters, indent=2))
    print("\nPictures per house:", ", ".join(f"{h} {totals[h]}" for h in HOUSES))
    print(f"Saved in {RAW_DIR}. Next step: python prepare.py")


if __name__ == "__main__":
    main()
