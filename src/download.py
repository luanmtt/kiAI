from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import threading
import shutil
import time
import json
import os


# --------------------------------------------------------------------------------------------------


BASE_URL = "https://osu.ppy.sh/api/v2"
OSU_FILE_URL = "https://osu.ppy.sh/osu"

MAX_WORKERS = 10
DELAY = 0.12  # 0.3 / 2.5

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


def _download_one(beatmap_id: int, out_path: Path, sem: threading.Semaphore, lock: threading.Lock,
                  stats: dict) -> tuple[int, bool, str]:
    """Baixa um único .osu file. Retorna (beatmap_id, success, error_msg)."""
    if out_path.exists() and out_path.stat().st_size > 0:
        with lock:
            stats["skipped"] += 1
        return beatmap_id, True, "skip"

    sem.acquire()
    try:
        r = requests.get(f"{OSU_FILE_URL}/{beatmap_id}", timeout=10)
        r.raise_for_status()

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(r.text)

        time.sleep(DELAY)

        with lock:
            stats["downloaded"] += 1
        return beatmap_id, True, "ok"

    except requests.exceptions.RequestException as e:
        with lock:
            stats["failed"] += 1
        return beatmap_id, False, str(e)

    finally:
        sem.release()


def download_osu_files(data_dir: str = "data", label_type: str | None = None):
    
    raw_dir = Path(data_dir) / "raw"
    if label_type:
        raw_dir = raw_dir / label_type

    all_ids = []
    for json_file in raw_dir.glob("*.json"):
        with open(json_file) as f:
            beatmaps = json.load(f)
        all_ids += [b["id"] for b in beatmaps]


    print(f"total entries : {len(all_ids)}")
    print(f"unique IDs    : {len(set(all_ids))}")
    print(f"duplicates    : {len(all_ids) - len(set(all_ids))}")
    

    osu_dir  = Path(data_dir) / "beatmaps"
    osu_dir.mkdir(parents=True, exist_ok=True)


    json_files = list(raw_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {raw_dir}/")
        return


    # count total maps across all files first
    all_maps = []
    for json_file in json_files:
        
        category = json_file.stem  # e.g. "stream" from "stream.json"
        
        with open(json_file) as f:
            beatmaps = json.load(f)
        
        for b in beatmaps:
            all_maps.append((category, b))


    total = len(all_maps)

    # filter out non-standard modes
    maps_to_download = [
        (cat, b) for cat, b in all_maps
        if b.get("mode") not in ["mania", "taiko", "fruits"]
    ]

    print(f"Found {total} maps across {len(json_files)} categories")
    print(f"Saving .osu files to {osu_dir}/")
    print(f"Downloading with {MAX_WORKERS} workers, {DELAY}s delay\n")

    sem = threading.Semaphore(MAX_WORKERS)
    lock = threading.Lock()
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for i, (category, b) in enumerate(maps_to_download):
            beatmap_id = b["id"]
            out_path   = osu_dir / f"{beatmap_id}.osu"
            fut = pool.submit(_download_one, beatmap_id, out_path, sem, lock, stats)
            futures[fut] = (i, category, b)

        for fut in as_completed(futures):
            i, category, b = futures[fut]
            beatmap_id, success, msg = fut.result()

            with lock:
                done = stats["downloaded"] + stats["skipped"] + stats["failed"]

            if done % 50 == 0 and done > 0:
                print(f"  [{done}/{len(maps_to_download)}] "
                      f"dl={stats['downloaded']} skip={stats['skipped']} fail={stats['failed']}")

            if not success and msg != "skip":
                print(f"  FAILED {beatmap_id}: {msg}")

            if done % 100 == 0 and done > 0:
                check_disk_space(osu_dir, done, len(maps_to_download))


# --------------------------------------------------------------------------------------------------
# download results:


    print(f"\nDone!")
    print(f"  downloaded : {stats['downloaded']}")
    print(f"  skipped    : {stats['skipped']} (already existed)")
    print(f"  failed     : {stats['failed']}")
    print(f"  total      : {len(maps_to_download)}")


# --------------------------------------------------------------------------------------------------
# main:


if __name__ == "__main__":
    download_osu_files()


# --------------------------------------------------------------------------------------------------
