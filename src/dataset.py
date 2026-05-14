from src.osu_parser import parse_and_feature

from pathlib import Path
import pandas as pd
import numpy as np
import json

# --------------------------------------------------------------------------------------------------

GENRE = {
    0: "any", 1: "unspecified", 2: "video_game", 3: "anime",
    4: "rock", 5: "pop", 6: "other", 7: "novelty",
    9: "hip_hop", 10: "electronic", 11: "metal",
    12: "classical", 13: "folk", 14: "jazz"
}


# --------------------------------------------------------------------------------------------------
# build dataset with both .osu and api-retrieved data.


def build_dataset(data_dir:str = "data") -> pd.DataFrame:
    
    raw_dir = Path(data_dir) / "raw"
    beatmap_dir = Path(data_dir) / "beatmaps"
    out_dir = Path(data_dir) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    
    # --------------------------------------------------------------------------------------------------


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

        print(f"Processing '{category}'...     → ({len(beatmaps)} maps)")

        for b in beatmaps:
            beatmap_id = b["id"]

            if beatmap_id in seen_ids:
                skipped_api += 1
                continue
            seen_ids.add(beatmap_id)

            bset = b.get("beatmapset", {})
            genre_id = bset.get("genre_id")
            
            ar_od_ratio = round(bset.get("ar") / bset.get("od"), 2) if bset.get("od") else 0

            row = {
                "label":                category,
                "id":                   beatmap_id,
                "title":                bset.get("title"),
                "artist":               bset.get("artist"),
                "creator":              bset.get("creator"),
                
                "total_length":         b.get("total_length"),
                "star_rating":          b.get("difficulty_rating"),
                "bpm":                  b.get("bpm"),
                "ar":                   b.get("ar"),
                "cs":                   b.get("cs"),
                "hit_length":           b.get("hit_length"),
                "od":                   b.get("accuracy"),
                
                "genre":                GENRE.get(genre_id, "unknown")
                
                #"drain":                b.get("drain"),
                
                # consigo com .osu.
                #"count_sliders":        b.get("count_sliders"),
                #"count_circles":        b.get("count_circles"),
                
                # Usar essa feature no futuro.
                #"tags":                 bset.get("tags", ""),
            }


            osu_path = beatmap_dir / f"{beatmap_id}.osu"

            if not osu_path.exists():
                skipped_osu += 1
                # still add the row, just with null osu features
                row.update({

                    "circles":              None,
                    "sliders":              None,
                    "spinners":             None,
                    "slider_ratio":         None,
                    "base_bpm":             None,
                    
                    "total_notes":          None,
                    "mean_distance":        None,
                    "std_distance":         None,
                    "stream_density":       None,
                        
                    #"kiai_section_count":  None,
                    "kiai_note_ratio":      None,
                    "kiai_mean_dist":       None,
                    "kiai_note_count":      None,
                    
                    "interval_variance":    None,
                    #"interval_cv":          None,

                    "mean_interval_ms":     None,
                    "notes_per_second":     None,
                    "mean_velocity":        None, 
                    "mean_sliders_length":    None,
                    
                    "rhythm_complexity":    None, 
                    "burst_density":        None,
                    "speed_index":          None,
                    "alt_ratio":            None,

                })

            else:
                try:
                    parsed = parse_and_feature(str(osu_path))
                    row.update(parsed)

                except Exception as e:
                    parse_errors += 1
                    print(f"  parse error on {beatmap_id}: {e}")
                    row.update({
                            
                        "circles":              None,
                        "sliders":              None,
                        "spinners":             None,
                        "slider_ratio":         None,
                        "base_bpm":             None,
                        
                        "total_notes":          None,
                        "mean_distance":        None,
                        "std_distance":         None,
                        "stream_density":       None,
                            
                        #"kiai_section_count":  None,
                        "kiai_note_ratio":      None,
                        "mean_kiai_dist":       None,
                        "kiai_note_count":      None,
                        
                        "interval_variance":    None,
                        #"interval_cv":          None,

                        "mean_interval_ms":     None,
                        "notes_per_second":     None,
                        "mean_velocity":        None, 
                        "mean_sliders_length":   None,
                        
                        "rhythm_complexity":    None,
                        "burst_density":        None,
                        "speed_index":          None,
                        "alt_ratio":            None,


                    })
                
            
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # introdução de algumas features relacionadas a AR, que não 
            row.update({
                "interval_variance_log":  round(np.log1p(row.get("interval_variance") or 0), 2),
                "ar_od_ratio": ar_od_ratio
            })
                   
            rows.append(row)


    # --------------------------------------------------------------------------------------------------
    # passagem para o df:


    df = pd.DataFrame(rows)
    
    out_path = out_dir / "dataset.csv"
    df.to_csv(out_path, index=False)

    print(f"\nDataset built:")
    print(f"  Total rows     : {len(df)}")
    print(f"  Skipped (dupe) : {skipped_api}")
    print(f"  Missing .osu   : {skipped_osu}")
    print(f"  Parse errors   : {parse_errors}")
    print(f"  Saved to       : {out_path}")
    #print(f"\nLabel distribution:")
    #print(df["label"].value_counts().to_string())

    return df

# --------------------------------------------------------------------------------------------------
