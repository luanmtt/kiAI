# kiAI — Data Processing & Collection

## Overview

This document covers the full data pipeline for the osu! beatmap classifier —
from fetching raw map data to building a clean feature matrix ready for training.

---

## 1. Authentication

The osu! API v2 uses OAuth 2.0. For public beatmap data, the **client credentials**
flow is sufficient — no user login is required.

You register an OAuth application on osu! settings and receive a **Client ID**
and **Client Secret**. These are exchanged for a temporary access token (valid
~24 hours) that is attached to every subsequent API request as a Bearer header.

The token must be refreshed for long scraping sessions.

---

## 2. Label Collection via osu!collector

Category labels are sourced from curated collections on[osu!collector](https://osucollector.com). Each collection is assumed to
represent a single map category (e.g. "stream maps", "farm maps").

### Category → Collection ID mapping

A dictionary maps each category name to its osu!collector collection ID:

```python
COLLECTIONS = {
    "stream":      1234,
    "farm":        5678,
    "jump":        9101,
    ...
}
```

The script iterates this dict, fetches all IDs per category, then saves the
results to `data/raw/{category}.json`.

---

## 3. Beatmap Metadata — osu! API v2

The osu! API v2 endpoint `GET /beatmaps` accepts up to **50 IDs per request**
using repeated `ids[]` query parameters. IDs are chunked into batches of 50
with a 0.3–0.5s delay between requests to avoid rate limiting.

### Fields retained per beatmap

| Field | Source | Notes |
|---|---|---|
| `id` | top-level | unique beatmap ID |
| `bpm` | top-level | |
| `ar` | top-level | approach rate |
| `cs` | top-level | circle size |
| `drain` | top-level | HP drain |
| `accuracy` | top-level | OD (overall difficulty) |
| `difficulty_rating` | top-level | star rating |
| `hit_length` | top-level | drain time in seconds |
| `total_length` | top-level | full length including breaks |
| `count_circles` | top-level | |
| `count_sliders` | top-level | |
| `count_spinners` | top-level | |
| `max_combo` | top-level | |
| `status` | top-level | ranked / loved / approved |
| `title`           | `beatmapset` | |
| `creator` | `beatmapset` | mapper username |
| `tags` | `beatmapset` | space-separated keywords |
| `genre_id` | `beatmapset` | integer, mapped to name |
| `title`, `artist` | `beatmapset` | song metadata |

Fields discarded: `failtimes`, `covers`, `checksum`, `preview_url`,
`nominations_summary`, `availability`, `ratings array`.

### Genre ID mapping

osu! stores music genre as an integer. It is mapped to a human-readable string
during dataset construction:

```
0: any       1: unspecified   2: video_game   3: anime
4: rock      5: pop           6: other        7: novelty
9: hip_hop  10: electronic   11: metal       12: classical
13: folk    14: jazz
```

Unknown or missing IDs fall back to `"unknown"`.

---

## 4. .osu File Download

The raw `.osu` file for each beatmap is downloaded from
`https://osu.ppy.sh/osu/{beatmap_id}`. These files contain the full hit object
data needed to compute spatial and timing features that are not available through
the API alone.

Files are saved to `data/beatmaps/{beatmap_id}.osu`.

### Safety measures

- **Skip check**: files are only skipped if they already exist **and** have
  `size > 0`. Zero-byte files from interrupted downloads are re-fetched.
- **Disk check every 100 maps**: prints current folder size and remaining free
  disk space. Raises a hard stop if free space drops below 500 MB.
- **Delay**: 0.3s between requests, increasing to 1s after any failure.
- **Re-runnable**: the script can be interrupted and restarted safely at any
  point — already-downloaded valid files are skipped.

---

## 5. Feature Extraction

Features are split into two tiers:

### Tier 1 — API features (no .osu file needed)

Extracted directly from the JSON responses. Fast, always available.

- BPM, AR, CS, OD, HP, star rating
- Hit length, total length
- Circle count, slider count, spinner count
- Slider ratio (`count_sliders / (count_circles + count_sliders)`)
- Max combo
- Genre (encoded)
- Tag keywords (binary-encoded: does the tag string contain "farm", "stream",
  "tech", etc.)

### Tier 2 — .osu file features

Parsed from the raw hit object and timing point data.

- **BPM from timing points**: more precise than the API value for variable-BPM maps
- **Kiai sections**: time ranges where the kiai flag is active
- **Kiai note ratio**: fraction of total notes that fall inside kiai sections
- **Mean note distance in kiai**: average Euclidean distance between consecutive
  hit objects during kiai, in osu! pixels (playfield is 512×384)
- **Note interval variance**: variance of time gaps between consecutive notes —
  high variance suggests tech or gimmick patterns; low variance suggests streams
  or consistency-focused maps
- **Stream density**: fraction of note intervals shorter than `60000 / (bpm * 4)`ms,
  indicating 1/4 beat streams

---

## 6. Dataset Construction

All per-map features are flattened into a single row and combined across all
category JSON files into one CSV:

```
data/processed/dataset.csv
```

Each row has all features plus a `label` column (the category name from the
collection dict key).

Duplicate map IDs (a map appearing in two collections) are flagged and dropped —
keeping the first occurrence.

---

## 7. Uncertainty & Label Noise

### Human-labeled data

Every label in this dataset originates from a human-curated osu!collector
collection. This introduces several sources of noise:

- **Curator bias**: the person who built a collection decides what counts as a
  "stream map" or "tech map". Two curators may disagree on borderline cases.
- **Community label drift**: terms like "farm map" or "aim-slop" are informal
  community slang. Their meaning shifts over time and varies between players.
- **Skill-level dependence**: a map may feel like a "jump map" to a 3-star player
  and a "speed map" to a 6-star player. The label reflects the curator's skill
  level and intent, not an objective property of the map.

### Multi-label reality

Most maps are genuinely multi-type. A map can be simultaneously a farm map and
a jump farm. By assigning a single label per collection, the dataset forces a
mutually exclusive framing onto a continuous, overlapping space. The model will
learn these boundaries as if they are hard — they are not.

This means the model's confidence scores should be interpreted as
**soft membership**, not certainty. A 70% "stream" / 25% "speed" output is
more informative than just the top-1 label.

### Collection quality variance

Some osu!collector collections are carefully curated; others are loosely themed
playlists. There is no programmatic way to distinguish them. A higher-quality
training set would require manual verification of a sample of each collection
before training.

### What the model cannot know

The model has no access to:
- How the map **feels** to play (feedback, flow, rhythm satisfaction)
- The mapper's stated intent
- Community consensus beyond what tags and collections approximate

It can only learn correlations between measurable features and the labels
provided. Any category that is defined more by feel than by measurable geometry
(e.g. "aim control", "precision") will have higher classification uncertainty
than categories with clear physical signatures (e.g. "stream", "slider map").

---

## Directory Structure

```
kiAI/
├── data/
│   ├── raw/                # one JSON per category from osu! API
│   ├── beatmaps/           # raw .osu files, named {beatmap_id}.osu
│   └── processed/
│       └── dataset.csv     # flat feature matrix with labels
├── src/
│   ├── auth.py             # OAuth token fetch
│   ├── fetch.py            # collection ID fetch + beatmap batch fetch
│   ├── download.py         # .osu file downloader with disk safety
│   ├── parser.py           # .osu file parser (hit objects, timing, kiai)
│   ├── features.py         # feature engineering
│   ├── dataset.py          # assembles dataset.csv
│   └── model.py            # Keras model definition
│
├── models/
│   └── model.keras
│
├── results/
│   ├── eda/                # has stuff, but WIP
│   ├── le.pkl
│   └── scaler.pkl
│
├── notebook.ipynb          # WIP        
│
└── README.md
```
