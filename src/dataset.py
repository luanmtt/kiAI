import json
from typing import dataclass_transform
import pandas as pd
from pathlib import Path
from osu_parser import parse_osu_file

# --------------------------------------------------------------------------------------------------

GENRE = {
    0: "any", 1: "unspecified", 2: "video_game", 3: "anime",
    4: "rock", 5: "pop", 6: "other", 7: "novelty",
    9: "hip_hop", 10: "electronic", 11: "metal",
    12: "classical", 13: "folk", 14: "jazz"
}


COLLECTIONS = {
    "stream":       "stream_maps.json",
    "farm":         "farm_maps.json",
    "speed":        "speed_maps.json",
    "aim_control":  "aim_control_maps.json", 
    "aim_slop":     "aim_slop_maps.json",
    "jump":         "jump_maps.json", 
    "gimmick":      "gimmick_maps.json",
    "consistency":  "consistency_maps.json",
    "precision":    "precision_maps.json",
    "stamina":      "stamina_maps.json",
    "alt":          "alt_maps.json",
    "tech":         "tech_maps.json",
    "slider":       "slider_maps.json",
    "reading":      "reading_maps.json",
    "finger_control":"finger_control_maps.json"
}   


# --------------------------------------------------------------------------------------------------
# build dataset with both .osu and api-retrieved data.

def build_dataset(data_dir:str = "data") -> pd.DataFrame:

    raw_dir = Path(data_dir) / "raw"
    beatmap_dir = Path(data_dir) / "beatmaps"
    out_dir = Path(data_dir) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(raw_dir.glob("*.json"))
    if not json_files:
        print("No JSON files found in data/raw/")
        return pd.DataFrame()
   
    rows = []
    skipped_api  = 0
    skipped_osu  = 0
    parse_errors = 0
    seen_ids     = set()

    for json_file in json_files:
        category = json_file.stem
        with open(json_file) as f:
            beatmaps = json.load(f)

        print(f"\nProcessing '{category}' ({len(beatmaps)} maps)...")

        for b in beatmaps:
            beatmap_id = b["id"]

            if beatmap_id in seen_ids:
                skipped_api += 1
                continue
            seen_ids.add(beatmap_id)

            bset = b.get("beatmapset", {})
            genre_id = bset.get("genre_id")

            row = {

                "label":                category,
                "id":                   beatmap_id,
                "bpm":                  b.get("bpm"),
                "ar":                   b.get("ar"),
                "cs":                   b.get("cs"),
                "length":               b.get("hit_length"),
                "drain":                b.get("drain"),
                "od":                   b.get("accuracy"),
                "difficulty_rating":    b.get("difficulty_rating"),
                "count_sliders":        b.get("count_sliders"),
                "count_circles":        b.get("count_circles"),
                "length":               b.get("total_length"),
                "title":                b.get("title"),
                "artist":               b.get("artist"),
                "creator":              bset.get("creator"),
                "tags":                 bset.get("tags", ""),
                "genre":                GENRE.get(genre_id, "unknown")
            }


    osu_path = beatmap_dir / f"{beatmap_id}.osu"

    if not osu_path.exists():
        skipped_osu += 1
        # still add the row, just with null osu features
        row.update({
            "base_bpm":           None,
            "kiai_section_count": None,
            "kiai_note_count":    None,
            "kiai_note_ratio":    None,
            "mean_kiai_dist":     None,
            "interval_variance":  None,
            "mean_interval_ms":   None,
            "stream_density":     None,
        })

    else:
        try:
            parsed = parse_osu_file(str(osu_path))
            row.update(parsed)

        except Exception as e:
            parse_errors += 1
            print(f"  parse error on {beatmap_id}: {e}")
            row.update({
                "base_bpm":           None,
                "kiai_section_count": None,
                "kiai_note_count":    None,
                "kiai_note_ratio":    None,
                "mean_kiai_dist":     None,
                "interval_variance":  None,
                "mean_interval_ms":   None,
                "stream_density":     None,
            })

        rows.append(row)


    df = pd.DataFrame(rows)
    
    out_path = out_dir / "dataset.csv"
    df.to_csv(out_path, index=False)

    print(f"\nDataset built:")
    print(f"  total rows     : {len(df)}")
    print(f"  skipped (dupe) : {skipped_api}")
    print(f"  missing .osu   : {skipped_osu}")
    print(f"  parse errors   : {parse_errors}")
    print(f"  saved to       : {out_path}")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    return df


# --------------------------------------------------------------------------------------------------
