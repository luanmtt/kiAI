# kiAI — Architecture

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                              │
│                                                                     │
│  main.py → user.py:run_train()                                      │
│  ├── check_collections_input()                                      │
│  │   └── data/collections_input.json                                │
│  │                                                                   │
│  ├── get_token() → auth.py                                          │
│  │                                                                   │
│  ├── For each category:                                             │
│  │   ├── get_ids_from_collector(id) → fetch.py                      │
│  │   │   └── GET osucollector.com/api/collections/{id}/beatmapsv2  │
│  │   │                                                               │
│  │   ├── get_beatmaps_bulk(ids, token) → fetch.py                   │
│  │   │   └── GET osu.ppy.sh/api/v2/beatmaps (batches of 50)        │
│  │   │                                                               │
│  │   └── Save → data/raw/{category}_maps.json                       │
│  │                                                                   │
│  └── download_osu_files() → download.py                             │
│      └── GET osu.ppy.sh/osu/{id} → data/beatmaps/{id}.osu          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATASET BUILD                                   │
│                                                                     │
│  build_dataset() → dataset.py                                       │
│  │                                                                   │
│  ├── For each {category}_maps.json in data/raw/:                    │
│  │   │                                                               │
│  │   ├── Read API metadata from JSON:                                │
│  │   │   bpm, ar, cs, od, star_rating, total_length, genre, ...     │
│  │   │                                                               │
│  │   ├── parse_and_feature(path) → osu_parser.py                    │
│  │   │   │                                                           │
│  │   │   ├── Parse .osu file into sections (TimingPoints, HitObjects)│
│  │   │   │                                                           │
│  │   │   └── feature_engineering(sections) → features.py             │
│  │   │       │                                                       │
│  │   │       ├── TimingPoints → base_bpm, kiai_ranges               │
│  │   │       ├── HitObjects → distances, intervals, hit type counts │
│  │   │       │                                                       │
│  │   │       └── Compute ~22 features:                               │
│  │   │           base_bpm, stream_density, burst_density,            │
│  │   │           mean_distance, std_distance,                        │
│  │   │           kiai_note_ratio, kiai_mean_dist,                    │
│  │   │           mean_interval_ms, interval_variance,                │
│  │   │           notes_per_second, mean_velocity,                    │
│  │   │           speed_index, alt_ratio, rhythm_complexity,          │
│  │   │           slider_ratio, mean_sliders_length                   │
│  │   │                                                               │
│  │   └── Merge API + .osu features → one row                        │
│  │                                                                   │
│  └── Save → data/processed/dataset.csv                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STATISTICAL ANALYSIS                              │
│                                                                     │
│  stats_analysis(df) → stat.py                                       │
│  │                                                                   │
│  ├── Per label × feature: mean, std, variance,                      │
│  │   skewness, kurtosis, normality test (Shapiro-Wilk)              │
│  │                                                                   │
│  └── Save → results/eda/eda_stats.csv                               │
│                                                                     │
│  build_expectations(stats_df, MOD_FEATURES) → stat.py               │
│  │                                                                   │
│  └── Build per-label feature stats for mod-affected features:       │
│      DT: {base_bpm, notes_per_second, mean_interval_ms}             │
│      HR: {ar, cs, od}                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MOD AUGMENTATION                                 │
│                                                                     │
│  apply_mods(df, stats_df) → stat.py                                 │
│  │                                                                   │
│  ├── For each row in df:                                            │
│  │   │                                                               │
│  │   ├── are_mods_needed(row, expectations) → stat.py               │
│  │   │   ├── If "dt" in label → return 0 (DT)                       │
│  │   │   ├── If "hr" in label → return 1 (HR)                       │
│  │   │   └── Statistical check:                                      │
│  │   │       For each mod feature, check if row is >1.5σ below      │
│  │   │       the label mean. If 2+ features flag → apply that mod.  │
│  │   │                                                               │
│  │   ├── If mod needed:                                              │
│  │   │   ├── to_dt(row, new_row) → mod.py                           │
│  │   │   │   ×1.5 base_bpm, bpm, notes_per_second, mean_velocity   │
│  │   │   │   ÷1.5 mean_interval_ms                                  │
│  │   │   │   ×0.67 total_length                                     │
│  │   │   │   perceived_ar_dt (nonlinear ms conversion)              │
│  │   │   │   perceived_od_dt (hit window ÷1.5)                      │
│  │   │   │   ar_od_ratio recalc                                      │
│  │   │   │   speed_index recalc (base_bpm × stream_density)         │
│  │   │   │                                                           │
│  │   │   └── to_hr(row, new_row) → mod.py                           │
│  │   │       ×1.3 cs, ×1.4 ar, ×1.4 od (all capped at 10)          │
│  │   │                                                               │
│  │   └── Append augmented row                                       │
│  │                                                                   │
│  └── Concat original df + augmented rows                             │
│                                                                     │
│  Save → data/processed/augmented.csv                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING                                     │
│                                                                     │
│  train() → model.py                                                 │
│  │                                                                   │
│  ├── Read augmented.csv                                             │
│  ├── Drop NaN rows                                                  │
│  ├── LabelEncoder → y (labels → integers)                           │
│  ├── StandardScaler → X (features → mean=0, std=1)                  │
│  ├── 80/20 split (stratified)                                       │
│  ├── Class weight balancing                                         │
│  │                                                                   │
│  ├── Model: Keras Sequential (256→128→64→softmax)                   │
│  │   AdamW, BatchNorm, Dropout, EarlyStopping                       │
│  │                                                                   │
│  ├── Evaluate: accuracy, classification report                      │
│  │                                                                   │
│  └── Save: models/model.keras, data/processed/scaler.pkl,           │
│             data/processed/le.pkl                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Mod Application Detail

### DT (Double Time)

| Feature | Transformation | Source |
|---|---|---|
| base_bpm | ×1.5, floor | osu!wiki |
| bpm | ×1.5, floor | osu!wiki |
| total_length | ×0.67, floor | 1/1.5 |
| mean_interval_ms | ÷1.5 | inverse speed |
| notes_per_second | ×1.5 | proportional to speed |
| mean_velocity | ×1.5 | proportional to speed |
| ar | perceived_ar_dt() | ms conversion chain |
| od | perceived_od_dt() | hit window ÷1.5 |
| ar_od_ratio | ar/od recalc | — |
| speed_index | base_bpm × stream_density | recalc after base_bpm change |
| stream_density | unchanged | invariant under uniform scaling |

### HR (Hard Rock)

| Feature | Transformation | Cap |
|---|---|---|
| cs | ×1.3 | 10 |
| ar | ×1.4 | 10 |
| od | ×1.4 | 10 |

## Known Issues

- `are_mods_needed`: DT statistical check is non-functional — `notes_per_second` and `mean_interval_ms` are not in the `features` list used by `stats_analysis`, so they never get stats. Only `base_bpm` can flag for DT, never reaching `MIN_FLAGS = 2`.
- `predict.py`: WIP — features list doesn't match `model.py`, mod application uses string comparison against integer return values.
- Precision maps always get HR via label shortcut, bypassing statistical check.
