import os
import time
import pandas as pd
from googleapiclient.discovery import build

# =========================
# CONFIGURATION
# =========================

API_KEY     = "AIzaSyATK-eMFHvvBMbWG1GjyxhxBmCO8GoVw9s"  # <- apna API key yahan daalo
TARGET_VIDEOS = 2000
OUTPUT_FILE   = "youtube_videos_dataset.csv"

# =========================
# SEARCH QUERIES
# pehle wale 20 + naye 20
# = 40 queries total
# = ~2000 videos
# =========================

SEARCH_QUERIES = [

    # ── PEHLE WALE 20 ──
    "technology review 2024",
    "cooking recipe tutorial",
    "travel vlog india",
    "fitness workout home",
    "study motivation students",
    "comedy funny moments",
    "news today hindi",
    "gaming highlights pc",
    "education science explained",
    "music cover song",
    "business tips entrepreneur",
    "cricket highlights match",
    "movie review bollywood",
    "dance tutorial beginners",
    "motivational speech hindi",
    "food street india",
    "diy project handmade",
    "car review drive",
    "fashion style outfit",
    "programming python tutorial",

    # ── NAYE 20 ──
    "health tips hindi",
    "stock market explained",
    "photography tutorial",
    "home decor ideas",
    "english speaking practice",
    "yoga meditation beginners",
    "mobile photography tips",
    "startup business india",
    "history facts interesting",
    "space science documentary",
    "budget travel india",
    "coding for beginners",
    "drawing painting tutorial",
    "parenting tips hindi",
    "finance money saving",
    "cricket analysis hindi",
    "bollywood behind scenes",
    "street food india",
    "wedding photography tips",
    "mental health awareness",

]

# =========================
# BUILD CLIENT
# =========================

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)

# =========================
# HELPER — check if short
# =========================

def is_short(duration_str):
    if not duration_str:
        return False
    import re
    hours   = int(re.search(r'(\d+)H', duration_str).group(1)) if 'H' in duration_str else 0
    minutes = int(re.search(r'(\d+)M', duration_str).group(1)) if 'M' in duration_str else 0
    seconds = int(re.search(r'(\d+)S', duration_str).group(1)) if 'S' in duration_str else 0
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds <= 65

# =========================
# HELPER — search IDs
# Cost: 100 units per call
# =========================

def search_video_ids(query, max_results=50):
    try:
        response = youtube.search().list(
            part="id",
            q=query,
            type="video",
            videoDuration="medium",
            maxResults=max_results,
            fields="items/id/videoId"
        ).execute()
        return [
            item["id"]["videoId"]
            for item in response.get("items", [])
            if "videoId" in item.get("id", {})
        ]
    except Exception as e:
        print(f"  Search error for '{query}': {e}")
        return []

# =========================
# HELPER — fetch stats
# Cost: 1 unit per 50 videos
# =========================

def fetch_video_stats(video_ids):
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        try:
            response = youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(batch),
                fields="items(id,snippet(title,publishedAt,tags),statistics(viewCount,likeCount,commentCount),contentDetails(duration))"
            ).execute()

            for item in response.get("items", []):
                snippet  = item.get("snippet", {})
                stats    = item.get("statistics", {})
                details  = item.get("contentDetails", {})

                title    = snippet.get("title", "")
                tags     = snippet.get("tags", [])
                pub      = snippet.get("publishedAt", "")
                duration = details.get("duration", "")

                if is_short(duration):
                    continue

                views    = int(stats.get("viewCount",    0))
                likes    = int(stats.get("likeCount",    0))
                comments = int(stats.get("commentCount", 0))

                if views == 0:
                    continue

                rows.append({
                    "title":         title,
                    "title_length":  len(title),
                    "hashtag_count": title.count("#"),
                    "tag_count":     len(tags),
                    "views":         views,
                    "likes":         likes,
                    "comments":      comments,
                    "publishedAt":   pub,
                    "duration":      duration,
                })

        except Exception as e:
            print(f"  Stats fetch error: {e}")

        time.sleep(0.3)

    return rows

# =========================
# LOAD EXISTING CSV
# (pehle ke 999 videos)
# =========================

existing_titles = set()
existing_rows   = []

if os.path.exists(OUTPUT_FILE):
    existing_df     = pd.read_csv(OUTPUT_FILE)
    existing_rows   = existing_df.to_dict("records")
    existing_titles = set(existing_df["title"].tolist())
    print(f"Existing dataset loaded: {len(existing_rows)} videos")
else:
    print("No existing dataset found — starting fresh")

# =========================
# MAIN COLLECTION LOOP
# =========================

all_rows   = list(existing_rows)   # start with existing 999
used_ids   = set()
unit_count = 0

already_have = len(all_rows)
still_needed = TARGET_VIDEOS - already_have

print("=" * 50)
print("YouTube Videos Dataset Collector")
print(f"Already have  : {already_have} videos")
print(f"Still needed  : {still_needed} more")
print(f"Target total  : {TARGET_VIDEOS} videos")
print("=" * 50)

if still_needed <= 0:
    print("\nTarget already reached! No collection needed.")
else:
    for query in SEARCH_QUERIES:

        if len(all_rows) >= TARGET_VIDEOS:
            break

        print(f"\n[{len(all_rows)}/{TARGET_VIDEOS}] Searching: '{query}'")

        # search — 100 units
        video_ids  = search_video_ids(query, max_results=50)
        unit_count += 100

        # filter already used IDs
        new_ids = [vid for vid in video_ids if vid not in used_ids]
        used_ids.update(new_ids)

        if not new_ids:
            print("  No new IDs found, skipping...")
            continue

        # fetch stats — 1 unit per batch
        rows = fetch_video_stats(new_ids)
        unit_count += max(1, len(new_ids) // 50)

        # filter duplicate titles with existing data
        new_rows = [r for r in rows if r["title"] not in existing_titles]
        existing_titles.update(r["title"] for r in new_rows)

        all_rows.extend(new_rows)

        print(f"  +{len(new_rows)} new videos | Total: {len(all_rows)} | Units: ~{unit_count}")

        time.sleep(1)

# =========================
# SAVE MERGED CSV
# =========================

print("\n" + "=" * 50)
final_df = pd.DataFrame(all_rows[:TARGET_VIDEOS])
final_df = final_df.drop_duplicates(subset=["title"])
final_df.to_csv(OUTPUT_FILE, index=False)

print(f"Total videos saved : {len(final_df)}")
print(f"Units used         : ~{unit_count}")
print(f"Saved to           : {OUTPUT_FILE}")
print("=" * 50)
print("\nSample:")
print(final_df.head())
