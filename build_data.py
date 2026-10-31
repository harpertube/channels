"""Builds the video data directory for the channels in channels.json.

Usage:
  YOUTUBE_API_KEY=... python build_data.py OUT_DIR --previous URL
  YOUTUBE_API_KEY=... python build_data.py OUT_DIR --fresh

Layout of OUT_DIR:
  index.json                       {"version", "channels": {handle: {"name", "manifest", "count"}}}
  <Handle>/manifest.<hash>.json    {"uploads": playlist ID, "skipped": [IDs of videos that can't be embedded],
                                    "batches": [batch file names, oldest first]}
  <Handle>/<NNNN>.<hash>.json      up to BATCH_SIZE videos, oldest first

Every file except index.json is named by a hash of its contents, so it never
changes once published and clients can cache it forever.

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

When run in GitHub Actions, it sets the step output "changed" to "true" or
"false", depending on whether the result differs from the previous data.
"""
import argparse
import hashlib
import json
import os
import re
import sys
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
RSS_ATTEMPTS = 3


def fail(message):
    sys.exit(f"error: {message}")


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
    # YouTube's feed server randomly answers many requests with a 404 or 500: about
    # 1 in 5 from a home connection, and far more often from GitHub's runners.
    errors = []
    for attempt in range(RSS_ATTEMPTS):
        if attempt:
            time.sleep(1)
        try:
            with urllib.request.urlopen(url, timeout=60) as res:
                root = ElementTree.parse(res).getroot()
            return [el.text for el in root.iter("{http://www.youtube.com/xml/schemas/2015}videoId")]
        except urllib.error.HTTPError as e:
            errors.append(f"HTTP {e.code}")
        except urllib.error.URLError as e:
            errors.append(str(e.reason))
    # The feed is only a quota-saving shortcut; the caller asks the API instead,
    # which gives the same answer for 1 unit. Logged so a feed that's broken for
    # good gets noticed.
    print(f"(RSS feed failed {RSS_ATTEMPTS} times: {', '.join(errors)}; checking with the API) ", end="", flush=True)
    return None


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
        fail(f"downloading {url} failed: HTTP {e.code}")
    except urllib.error.URLError as e:
        fail(f"downloading {url} failed: {e.reason}")


def load_previous_index(base_url):
    """Returns (index, its raw bytes) of the previously published data."""
    url = base_url + "index.json"
    try:
        with urllib.request.urlopen(url, timeout=60) as res:
            raw = res.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            fail(f"no previous data at {url}. If nothing has been published yet, use --fresh "
                 "(in GitHub, run the workflow by hand with 'fresh' checked).")
        fail(f"downloading {url} failed: HTTP {e.code}")
    except urllib.error.URLError as e:
        fail(f"downloading {url} failed: {e.reason}")
    index = json.loads(raw)
    if index.get("version") != FORMAT_VERSION:
        fail(f"{url} is format version {index.get('version')}, expected {FORMAT_VERSION}. Rebuild with --fresh.")
    return index, raw


def resolve_uploads(key, handle):
    res = yt_api(key, "channels", part="contentDetails", forHandle=handle)
    if not res.get("items"):
        fail(f"YouTube channel not found: {handle}")
    return res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def fetch_new_videos(key, uploads, known):
    """Returns (playable new videos oldest first, IDs of new videos that can't be embedded).

    A video is new if it's not in `known`. The uploads playlist is newest
    first, so this pages through it until it reaches a page containing a video
    it already has.
    """
    if known:
        rss_ids = fetch_rss_ids(uploads)
        if rss_ids is not None and all(i in known for i in rss_ids):
            return [], []

    new_ids = []
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
    new_ids = list(dict.fromkeys(new_ids))

    videos = {}
    not_embeddable = []
    for start in range(0, len(new_ids), PAGE_SIZE):
        res = yt_api(key, "videos", part="snippet,contentDetails,status", id=",".join(new_ids[start:start + PAGE_SIZE]))
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
    return [videos[i] for i in reversed(new_ids) if i in videos], not_embeddable


def update_channel(key, previous_base, handle, previous_entry):
    """Returns (manifest without its batch names, batches, number of videos added)."""
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
    known.update(skipped)
    new_videos, not_embeddable = fetch_new_videos(key, uploads, known)
    for v in new_videos:
        if not batches or len(batches[-1]) >= BATCH_SIZE:
            batches.append([])
        batches[-1].append(v)
    return {"uploads": uploads, "skipped": skipped + not_embeddable}, batches, len(new_videos)


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
    if args.previous:
        previous_base = args.previous if args.previous.endswith("/") else args.previous + "/"
        previous_index, previous_index_raw = load_previous_index(previous_base)
        previous_channels = previous_index["channels"]

    channels = json.loads((Path(__file__).parent / "channels.json").read_text(encoding="utf-8"))
    handles = [ch["handle"].lower() for ch in channels]
    if len(set(handles)) != len(handles):
        fail("channels.json lists the same handle more than once")

    files = {}
    index = {}
    for ch in channels:
        handle = ch["handle"]
        # Printed first so an error while updating the channel shows which channel it was.
        print(f"{handle}: ", end="", flush=True)
        manifest, batches, added = update_channel(key, previous_base, handle, previous_channels.get(handle))

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

    index_raw = (json.dumps({"version": FORMAT_VERSION, "channels": index}, ensure_ascii=False, indent=1) + "\n").encode()
    files["index.json"] = index_raw
    for rel, data in files.items():
        path = args.out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    # index.json names every other file by its content hash, so it's unchanged
    # exactly when nothing is.
    changed = index_raw != previous_index_raw
    print("Changed from the previous data." if changed else "Unchanged from the previous data.")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"changed={'true' if changed else 'false'}\n")


if __name__ == "__main__":
    main()
