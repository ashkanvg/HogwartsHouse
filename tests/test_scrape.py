"""The picture scraper's Commons logic, with made-up API answers (no internet needed)."""
import scrape


def test_names_written_differently_still_match():
    assert scrape.plain_name("Dame Maggie Smith") == "maggie smith"
    assert scrape.plain_name("Sir Michael Gambon") == "michael gambon"
    assert scrape.plain_name("Helena Bonham-Carter") == "helena bonham carter"


def test_finding_an_actors_commons_category_on_wikidata():
    assert scrape.commons_category("Daniel Radcliffe") == "Daniel Radcliffe"
    link = "https://commons.wikimedia.org/wiki/Category:Mark_Williams_(actor)"
    assert scrape.commons_category(None, link) == "Mark Williams (actor)"
    assert scrape.commons_category(None, "https://commons.wikimedia.org/wiki/Emma_Watson") is None  # a gallery
    entity = {"claims": {"P373": [{"mainsnak": {"datavalue": {"value": "Maggie Smith"}}}]}}
    assert scrape.entity_category(entity) == "Maggie Smith"
    assert scrape.entity_category({"sitelinks": {"commonswiki": {"title": "Category:Joshua Herdman"}}}) == "Joshua Herdman"
    assert scrape.entity_category({}) is None


def test_a_name_search_only_picks_british_or_irish_actors():
    assert scrape.BRITISH_ACTOR.search("English actress (1934–2024)")
    assert scrape.BRITISH_ACTOR.search("Northern Irish actor")
    assert not scrape.BRITISH_ACTOR.search("American quiltmaker")
    assert not scrape.BRITISH_ACTOR.search("English footballer")


def page(title, index=1, licence="CC BY-SA 2.0", mime="image/jpeg"):
    """A Commons search result, the way the API returns it."""
    return {"title": title, "index": index, "imageinfo": [{
        "mime": mime, "url": f"https://upload/{title}", "thumburl": f"https://thumb/{title}",
        "descriptionurl": f"https://commons/{title}",
        "extmetadata": {"LicenseShortName": {"value": licence}, "Artist": {"value": "<a href='x'>Jo Photo</a>"}}}]}


def test_only_free_photos_of_just_this_person_are_kept():
    pages = [
        page("File:Daniel Radcliffe 2011.jpg", index=2),
        page("File:Daniel Radcliffe at Comic-Con.jpg", index=1),
        page("File:Daniel Radcliffe wax figure.jpg"),               # a wax figure
        page("File:Signature of Daniel Radcliffe.png"),             # a signature
        page("File:Daniel Radcliffe & Joe Hill.jpg"),               # two people
        page("File:Daniel Radcliffe, Emma Watson.jpg"),             # another actor from the films
        page("File:Daniel Radcliffe 2009.jpg", licence="Fair use"),  # not a free licence
        page("File:Daniel Radcliffe.tif", mime="image/tiff"),       # not a JPEG or PNG
    ]
    photos = scrape.usable_photos(pages, other_actors={"emma watson"})
    assert [p["page"] for p in photos] == ["https://commons/File:Daniel Radcliffe at Comic-Con.jpg",
                                           "https://commons/File:Daniel Radcliffe 2011.jpg"]  # best match first
    assert photos[0]["url"].startswith("https://thumb/")
    assert photos[0]["author"] == "Jo Photo" and photos[0]["licence"] == "CC BY-SA 2.0"


def test_actor_photos_are_topped_up_without_downloading_twice(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image

    monkeypatch.setattr(scrape, "RAW_DIR", tmp_path)
    monkeypatch.setattr(scrape.time, "sleep", lambda seconds: None)  # no waiting in tests
    photos = [{"url": f"https://thumb/{i}", "page": f"https://commons/{i}", "author": "A", "licence": "CC BY 2.0"}
              for i in range(6)]
    monkeypatch.setattr(scrape, "commons_photos", lambda category, limit, others: photos)
    downloaded, busy = [], {1}  # photo 1 fails the first time: Wikimedia says it's too busy

    def fake_download(url):
        number = int(url.rsplit("/", 1)[1])
        downloaded.append(number)
        if number in busy:
            busy.discard(number)
            return None, True
        noise = np.random.default_rng(number).integers(0, 255, (120, 120, 3), dtype=np.uint8)
        return Image.fromarray(noise), False  # a different picture for every photo

    monkeypatch.setattr(scrape, "download_commons_photo", fake_download)
    character = {"name": "Hermione Granger", "house": "Gryffindor"}

    assert scrape.scrape_actor_photos(character, "Emma Watson", 3, set())[0] == 3
    assert downloaded == [0, 1, 2, 3]  # photo 1 failed, so it took four tries to get three
    downloaded.clear()
    assert scrape.scrape_actor_photos(character, "Emma Watson", 5, set())[0] == 5  # a later run tops up
    assert downloaded == [1, 4]  # photos 0, 2 and 3 are not downloaded again
