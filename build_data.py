"""Builds the video data directory for the channels in channels.json.

Usage:
  YOUTUBE_API_KEY=... python build_data.py OUT_DIR --previous URL
  YOUTUBE_API_KEY=... python build_data.py OUT_DIR --fresh

Layout of OUT_DIR:
  index.json                       {"version", "channels": {handle: {"name", "manifest", "count"}}}
  <Handle>/manifest.<hash>.json    {"uploads": playlist ID, "skipped": [IDs of videos that can't be embedded],
                                    "batches": [batch file names, oldest first]}
  <Handle>/<NNNN>.<hash>.json      up to BATCH_SIZE videos, oldest first
  shorts.json                      {handle: {"shorts": [IDs of Shorts],
                                    "unchecked": [IDs of videos the Shorts check couldn't check yet, oldest first]}}

Every file except index.json and shorts.json is named by a hash of its
contents, so it never changes once published and clients can cache it forever.
shorts.json is only for this script; clients never need it.

Each channel is an append-only log: new uploads are appended to the last batch
until it holds BATCH_SIZE videos, then a new batch starts. Full batches are
never rewritten, even if one of their videos is later deleted, so a client
never downloads them twice.

The previously published data (--previous, the base URL of the deployed site)
is this script's memory: it's downloaded, extended with any uploads not seen
before, and written to OUT_DIR in full. A channel not in the previous data
(or every channel, with --fresh) gets its whole upload history backfilled,
which costs about 2 API quota units per 50 videos. A channel removed from
channels.json is left out.

Keeping a channel up to date is nearly free: its RSS feed (no quota) is checked
first, and the API is only called when the feed shows a video not seen before
or the feed can't be downloaded. Then it costs 1 unit per 50 playlist entries
paged through, plus 1 unit per 50 new videos looked up.

Shorts are left out. The API can't tell them apart, so each new video short
enough to be one is checked on youtube.com, which costs no quota, and the
result is remembered in shorts.json: a Short goes in its "shorts" list and any
other video in a batch. If YouTube rate limits the check, the video goes in
"unchecked" instead and is checked again on the next run, which costs 1 unit
per 50 such videos to look them up again. Any other failure of the check fails
the run, since it means the check may have stopped working.

When run in GitHub Actions, it sets the step output "changed" to "true" or
"false", depending on whether the result differs from the previous data.
"""
import argparse
import collections
import hashlib
import html
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree

API = "https://www.googleapis.com/youtube/v3/"
FORMAT_VERSION = 1
BATCH_SIZE = 100
# The most items the YouTube API returns per call.
PAGE_SIZE = 50
# Tries per request to youtube.com's website (the RSS feed and Shorts check).
WEB_ATTEMPTS = 3
# Shorts can be at most 3 minutes long, so only videos up to this length are
# checked. A few seconds of margin in case YouTube rounds a duration up.
SHORTS_MAX_SECONDS = 185


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


# Reports a redirect as an HTTPError instead of following it.
no_redirect_opener = urllib.request.build_opener(_NoRedirect)
# Set once YouTube rate limits a Shorts check.
rate_limited = threading.Event()


def fail(message):
    sys.exit(f"error: {message}")


def describe_http_error(e):
    """Describes an HTTPError with what the server said about it, not just the status code."""
    body = e.read().decode(errors="replace")
    if "html" in (e.headers.get("Content-Type") or ""):
        body = re.sub(r"(?s)<(style|script)[^>]*>.*?</\1>", "", body)
        body = html.unescape(re.sub(r"<[^>]+>", " ", body))
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) > 300:
        body = body[:300] + "..."
    return f'HTTP {e.code} from "{e.headers.get("Server", "unknown server")}": {body or "(empty body)"}'


def yt_api(key, endpoint, **params):
    url = API + endpoint + "?" + urllib.parse.urlencode({**params, "key": key})
    try:
        with urllib.request.urlopen(url, timeout=60) as res:
            return json.load(res)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        fail(f'YouTube API "{endpoint}" failed (HTTP {e.code}) with params {params}:\n{body}')


def fetch_rss_ids(uploads):
    """Returns the IDs of a channel's newest uploads (about 15) from its RSS feed, which costs no API quota.

    Returns None if the feed can't be downloaded.
    """
    # An uploads playlist ID is the channel ID with "UC" replaced by "UU".
    url = "https://www.youtube.com/feeds/videos.xml?channel_id=UC" + uploads.removeprefix("UU")
    # YouTube's feed server often answers valid channels with its generic "404.
    # That's an error." page or a 500, giving no reason: about 1 in 5 requests
    # from a home connection early on 2026-10-31, and most requests a few hours
    # later, from GitHub's runners and home alike.
    errors = []
    for attempt in range(WEB_ATTEMPTS):
        if attempt:
            time.sleep(1)
        try:
            with urllib.request.urlopen(url, timeout=60) as res:
                root = ElementTree.parse(res).getroot()
            return [el.text for el in root.iter("{http://www.youtube.com/xml/schemas/2015}videoId")]
        except urllib.error.HTTPError as e:
            errors.append(describe_http_error(e))
        except urllib.error.URLError as e:
            errors.append(str(e.reason))
    # The feed is only a quota-saving shortcut; the caller asks the API instead,
    # which gives the same answer for 1 unit. Logged so a feed that's broken for
    # good gets noticed. Goes to stderr so it stays out of the data branch's
    # commit messages, which are built from stdout.
    print(f"\nRSS feed {url} failed {WEB_ATTEMPTS} times; checking with the API instead:", file=sys.stderr)
    for error, count in collections.Counter(errors).items():
        print(f"  {count}x {error}", file=sys.stderr)
    return None


def is_short(video_id):
    """Returns whether a video is a YouTube Short, or None if YouTube is rate limiting the check.

    The API has no way to tell. youtube.com/shorts/ID shows a Short (200) and
    redirects any other video to its /watch page (303). Costs no API quota.
    Once YouTube rate limits one check, the rest of the run's checks return
    None without asking.
    """
    if rate_limited.is_set():
        return None
    url = "https://www.youtube.com/shorts/" + video_id
    errors = []
    for attempt in range(WEB_ATTEMPTS):
        if attempt:
            time.sleep(1)
        try:
            with no_redirect_opener.open(urllib.request.Request(url, method="HEAD"), timeout=60) as res:
                if res.status == 200:
                    return True
                errors.append(f"HTTP {res.status}")
        except urllib.error.HTTPError as e:
            location = e.headers.get("Location") or ""
            if e.code == 303 and location.split("&")[0] == "https://www.youtube.com/watch?v=" + video_id:
                return False
            # YouTube's two known ways of saying "too many requests".
            if e.code == 429 or (e.code in (302, 303) and location.startswith("https://www.google.com/sorry/")):
                rate_limited.set()
                return None
            errors.append(f"HTTP {e.code}" + (f" redirecting to {location}" if location else ""))
        except urllib.error.URLError as e:
            errors.append(str(e.reason))
    # Anything else means the check itself may be broken, so the run fails
    # rather than publishing a guess.
    fail(f"Shorts check {url} failed {WEB_ATTEMPTS} times: {'; '.join(errors)}")


def parse_duration(iso):
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        fail(f"unexpected video duration format from YouTube: {iso!r}")
    d, h, mins, s = (int(x or 0) for x in m.groups())
    return ((d * 24 + h) * 60 + mins) * 60 + s


def channel_dir(handle):
    return handle.removeprefix("@")


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


def hashed_name(stem, data):
    return f"{stem}.{hashlib.sha256(data).hexdigest()[:12]}.json"


def download(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as res:
            return res.read()
    except urllib.error.HTTPError as e:
        fail(f"downloading {url} failed: {describe_http_error(e)}")
    except urllib.error.URLError as e:
        fail(f"downloading {url} failed: {e.reason}")


def download_previous(url):
    """Downloads a file that every build publishes, so its absence means the data needs rebuilding."""
    try:
        with urllib.request.urlopen(url, timeout=60) as res:
            return res.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            fail(f"no previous data at {url} ({describe_http_error(e)}). If nothing has been published yet, "
                 "or it was published before this file existed, use --fresh "
                 "(in GitHub, run the workflow by hand with 'fresh' checked).")
        fail(f"downloading {url} failed: {describe_http_error(e)}")
    except urllib.error.URLError as e:
        fail(f"downloading {url} failed: {e.reason}")


def load_previous_index(base_url):
    """Returns (index, its raw bytes) of the previously published data."""
    url = base_url + "index.json"
    raw = download_previous(url)
    index = json.loads(raw)
    if index.get("version") != FORMAT_VERSION:
        fail(f"{url} is format version {index.get('version')}, expected {FORMAT_VERSION}. Rebuild with --fresh.")
    return index, raw


def resolve_uploads(key, handle):
    res = yt_api(key, "channels", part="contentDetails", forHandle=handle)
    if not res.get("items"):
        fail(f"YouTube channel not found: {handle}")
    return res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def fetch_new_videos(key, uploads, known, recheck):
    """Returns (playable videos oldest first, IDs of videos that can't be embedded).

    Looks up the videos in `recheck` (oldest first) and the new ones. A video is
    new if it's not in `known`. The uploads playlist is newest first, so this
    pages through it until it reaches a page containing a video it already has.
    """
    new_ids = []
    rss_ids = fetch_rss_ids(uploads) if known else None
    if rss_ids is None or any(i not in known for i in rss_ids):
        page_token = None
        while True:
            params = {"part": "contentDetails", "playlistId": uploads, "maxResults": PAGE_SIZE}
            if page_token:
                params["pageToken"] = page_token
            page = yt_api(key, "playlistItems", **params)
            ids = [item["contentDetails"]["videoId"] for item in page["items"]]
            new_ids.extend(i for i in ids if i not in known)
            page_token = page.get("nextPageToken")
            if not page_token or any(i in known for i in ids):
                break
    # The videos to recheck are older than any new one.
    lookup_ids = list(dict.fromkeys(recheck + new_ids[::-1]))

    videos = {}
    not_embeddable = []
    for start in range(0, len(lookup_ids), PAGE_SIZE):
        res = yt_api(key, "videos", part="snippet,contentDetails,status", id=",".join(lookup_ids[start:start + PAGE_SIZE]))
        for v in res["items"]:
            if not v["status"]["embeddable"]:
                not_embeddable.append(v["id"])
            # Live and upcoming streams aren't recorded anywhere, so they're picked
            # up on a later run once they've become regular videos.
            elif v["snippet"]["liveBroadcastContent"] == "none":
                videos[v["id"]] = {
                    "id": v["id"],
                    "title": v["snippet"]["title"],
                    "seconds": parse_duration(v["contentDetails"]["duration"]),
                }
    return [videos[i] for i in lookup_ids if i in videos], not_embeddable


def update_channel(key, previous_base, handle, previous_entry, previous_shorts):
    """Returns (manifest without its batch names, batches, number of videos added, shorts.json entry)."""
    shorts = previous_shorts.get("shorts", [])
    unchecked = previous_shorts.get("unchecked", [])
    if previous_entry:
        manifest = json.loads(download(previous_base + previous_entry["manifest"]))
        uploads = manifest["uploads"]
        skipped = manifest["skipped"]
        dir_url = previous_base + channel_dir(handle) + "/"
        with ThreadPoolExecutor(16) as pool:
            batches = [json.loads(b) for b in pool.map(lambda name: download(dir_url + name), manifest["batches"])]
    else:
        uploads = resolve_uploads(key, handle)
        skipped = []
        batches = []

    known = {v["id"] for batch in batches for v in batch}
    known.update(skipped, shorts, unchecked)
    # A video that was unchecked but is gone now (deleted, or made private) is
    # simply not returned, so it's dropped.
    videos, not_embeddable = fetch_new_videos(key, uploads, known, unchecked)

    candidates = [v["id"] for v in videos if v["seconds"] <= SHORTS_MAX_SECONDS]
    with ThreadPoolExecutor(8) as pool:
        short = dict(zip(candidates, pool.map(is_short, candidates)))
    unchecked = []
    added = 0
    for v in videos:
        # Videos too long to be Shorts weren't checked.
        state = short.get(v["id"], False)
        if state is None:
            unchecked.append(v["id"])
        elif state:
            shorts.append(v["id"])
        else:
            if not batches or len(batches[-1]) >= BATCH_SIZE:
                batches.append([])
            batches[-1].append(v)
            added += 1
    manifest = {"uploads": uploads, "skipped": skipped + not_embeddable}
    return manifest, batches, added, {"shorts": shorts, "unchecked": unchecked}


def main():
    parser = argparse.ArgumentParser(description="Build the video data directory for the channels in channels.json.")
    parser.add_argument("out_dir", type=Path, help="directory to write; must be empty or not exist")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--previous", metavar="URL", help="base URL of the previously published data")
    source.add_argument("--fresh", action="store_true", help="ignore previous data and backfill every channel")
    args = parser.parse_args()

    key = os.environ.get("YOUTUBE_API_KEY") or fail("the YOUTUBE_API_KEY environment variable is not set")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        fail(f"output directory {args.out_dir} is not empty")

    previous_base = None
    previous_channels = {}
    previous_index_raw = None
    previous_shorts = {}
    previous_shorts_raw = None
    if args.previous:
        previous_base = args.previous if args.previous.endswith("/") else args.previous + "/"
        previous_index, previous_index_raw = load_previous_index(previous_base)
        previous_channels = previous_index["channels"]
        previous_shorts_raw = download_previous(previous_base + "shorts.json")
        previous_shorts = json.loads(previous_shorts_raw)

    channels = json.loads((Path(__file__).parent / "channels.json").read_text(encoding="utf-8"))
    handles = [ch["handle"].lower() for ch in channels]
    if len(set(handles)) != len(handles):
        fail("channels.json lists the same handle more than once")

    files = {}
    index = {}
    shorts = {}
    for ch in channels:
        handle = ch["handle"]
        # Printed first so an error while updating the channel shows which channel it was.
        print(f"{handle}: ", end="", flush=True)
        manifest, batches, added, shorts[handle] = update_channel(
            key, previous_base, handle, previous_channels.get(handle), previous_shorts.get(handle, {}))

        names = []
        for i, batch in enumerate(batches):
            data = dumps(batch)
            name = hashed_name(f"{i:04d}", data)
            files[f"{channel_dir(handle)}/{name}"] = data
            names.append(name)
        manifest_data = dumps({**manifest, "batches": names})
        manifest_path = f"{channel_dir(handle)}/{hashed_name('manifest', manifest_data)}"
        files[manifest_path] = manifest_data

        count = sum(map(len, batches))
        index[handle] = {"name": ch["name"], "manifest": manifest_path, "count": count}
        print(f"{count} videos ({added} new) in {len(batches)} batches", flush=True)
        if shorts[handle]["unchecked"]:
            # stderr, to stay out of the data branch's commit messages.
            print(f"{handle}: YouTube is rate limiting the Shorts check; "
                  f"{len(shorts[handle]['unchecked'])} videos left for a later run.", file=sys.stderr, flush=True)

    index_raw = (json.dumps({"version": FORMAT_VERSION, "channels": index}, ensure_ascii=False, indent=1) + "\n").encode()
    files["index.json"] = index_raw
    shorts_raw = (json.dumps(shorts, indent=1) + "\n").encode()
    files["shorts.json"] = shorts_raw
    for rel, data in files.items():
        path = args.out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    # index.json names every other file by its content hash, so it and
    # shorts.json are unchanged exactly when nothing is.
    changed = index_raw != previous_index_raw or shorts_raw != previous_shorts_raw
    print("Changed from the previous data." if changed else "Unchanged from the previous data.")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"changed={'true' if changed else 'false'}\n")


if __name__ == "__main__":
    main()
