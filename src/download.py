from pathlib import Path
import requests
import shutil
import time
import json
import os


# --------------------------------------------------------------------------------------------------


BASE_URL = "https://osu.ppy.sh/api/v2"
OSU_FILE_URL = "https://osu.ppy.sh/osu"

GENRE = {
    0: "any", 1: "unspecified", 2: "video_game", 3: "anime",
    4: "rock", 5: "pop", 6: "other", 7: "novelty",
    9: "hip_hop", 10: "electronic", 11: "metal",
    12: "classical", 13: "folk", 14: "jazz"
}


# --------------------------------------------------------------------------------------------------
# memory checks every 100 iterations.


def get_dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def check_disk_space(osu_dir: Path, i: int, total: int):
    if i % 100 == 0:
        dir_mb   = get_dir_size_mb(osu_dir)
        free_mb  = shutil.disk_usage(osu_dir).free / (1024 * 1024)
        print(f"\n  [{i}/{total}] disk check:")
        print(f"    data/beatmaps size : {dir_mb:.1f} MB")
        print(f"    free disk space    : {free_mb:.0f} MB")
        if free_mb < 500:
            raise RuntimeError(f"Low disk space: only {free_mb:.0f} MB remaining. Stopping.")


# --------------------------------------------------------------------------------------------------
# download:


def download_osu_files(data_dir: str = "data"):
    
    all_ids = []
    for json_file in Path("data/raw").glob("*.json"):
        with open(json_file) as f:
            beatmaps = json.load(f)
        all_ids += [b["id"] for b in beatmaps]


    print(f"total entries : {len(all_ids)}")
    print(f"unique IDs    : {len(set(all_ids))}")
    print(f"duplicates    : {len(all_ids) - len(set(all_ids))}")
    

    raw_dir  = Path(data_dir) / "raw"
    osu_dir  = Path(data_dir) / "beatmaps"
    osu_dir.mkdir(parents=True, exist_ok=True)


    json_files = list(raw_dir.glob("*.json"))
    if not json_files:
        print("No JSON files found in data/raw/")
        return


    # count total maps across all files first
    all_maps = []
    for json_file in json_files:
        
        category = json_file.stem  # e.g. "stream" from "stream.json"
        
        with open(json_file) as f:
            beatmaps = json.load(f)
        
        for b in beatmaps:
            all_maps.append((category, b))


    total      = len(all_maps)
    skipped    = 0
    downloaded = 0
    failed     = 0

    print(f"Found {total} maps across {len(json_files)} categories")
    print(f"Saving .osu files to {osu_dir}/\n")

    for i, (category, b) in enumerate(all_maps, 1):
        beatmap_id = b["id"]
        out_path   = osu_dir / f"{beatmap_id}.osu"

        if b["mode"] in ["mania", "taiko", "fruits"]:
            continue

        # skip if already downloaded (safe to re-run)
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            if skipped % 50 == 0:
                print(f"  [{i}/{total}] skipped {skipped} already-downloaded files...")
            continue

        try:
            r = requests.get(f"{OSU_FILE_URL}/{beatmap_id}", timeout=10)
            r.raise_for_status()

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(r.text)

            downloaded += 1

            # progress print every 25 downloads
            if downloaded % 25 == 0:
                genre_name = GENRE.get(b.get("beatmapset", {}).get("genre_id"), "unknown")
                print(f"  [{i}/{total}] downloaded {downloaded} so far | "
                      f"last: {beatmap_id} ({category}, {genre_name})")

            time.sleep(0.3)  # polite delay — osu! will rate limit you without this
            
            if downloaded % 100 == 0: 
                check_disk_space(osu_dir, i, total)

        except requests.exceptions.RequestException as e:
            failed += 1
            print(f"  [{i}/{total}] FAILED {beatmap_id}: {e}")
            time.sleep(1)


# --------------------------------------------------------------------------------------------------
# download results:


    print(f"\nDone!")
    print(f"  downloaded : {downloaded}")
    print(f"  skipped    : {skipped} (already existed)")
    print(f"  failed     : {failed}")
    print(f"  total      : {total}")


# --------------------------------------------------------------------------------------------------
# main:


if __name__ == "__main__":
    download_osu_files()


# --------------------------------------------------------------------------------------------------
