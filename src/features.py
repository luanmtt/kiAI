import math


# --------------------------------------------------------------------------------------------------
# feature engineering para complementar dataset


def feature_engineering(sections: dict) -> dict:
    
     
    # --------------------------------------------------------------------------------------------------
    # kiai info:


    kiai_ranges = []    # aqui, temos como targets: start_ms, end_ms
    base_bpm = None
    kiai_on = None

    for line in sections.get("TimingPoints", []):

        parts = line.split(",")
        #print(f'Parts count: {len(parts)} | raw: {line}')

        if len(parts) < 8:
            continue

        time        = float(parts[0])
        beat_length = float(parts[1])
        uninherited = int(parts[6])
        effects     = int(parts[7])
        kiai_active = bool(effects & 1)

        #print(f'beat-length: {beat_length:.4f}')
        
        if uninherited == 1 and beat_length > 0:

            bpm = 60000 / beat_length

            if base_bpm is None:
                base_bpm = bpm # pegando o bpm dominante com essa lógica
                #print(f'bpm {base_bpm}\n')
            
        if kiai_active and kiai_on is None:
            kiai_on = time

        elif not kiai_active and kiai_on is not None:
            kiai_ranges.append((kiai_on, time))
            kiai_on = None

    if kiai_on is not None:
        kiai_ranges.append((kiai_on, float("inf")))


    # --------------------------------------------------------------------------------------------------
    # hit objects:


    hit_obj = [] # esse target é uma lista de (x,y,time_ms, type_int)
    
    for line in sections.get("HitObjects", []):

        parts = line.split(",")

        if len(parts) < 4:
            continue

        x   = float(parts[0])
        y   = float(parts[1])
        t   = float(parts[2])
        typ = int(parts[3])
        
        hit_obj.append((x,y,t,typ))

    total_notes = len(hit_obj)

    

    def in_kiai(t):
        return any(start <= t <= end for start, end in kiai_ranges)
    
    kiai_obj = [(x,y,t) for x,y,t, _ in hit_obj if in_kiai(t)]


    # --------------------------------------------------------------------------------------------------
    # distâncias entre objetos enquanto o kiai é ativo.


    kiai_distances = []
    for i in range(1, len(kiai_obj)):

        k_dx = kiai_obj[i][0] - kiai_obj[i-1][0]
        k_dy = kiai_obj[i][1] - kiai_obj[i-1][1]
        kiai_distances.append(math.sqrt(k_dx**2 + k_dy**2)) 


    mean_kiai_distance = (
        sum(kiai_distances) / len(kiai_distances) if kiai_distances else 0.0
    )


    # --------------------------------------------------------------------------------------------------
    # análise estatística de todas as distâncias do mapa:
    

    all_distances = []
    for i in range(1, len(hit_obj)):

        dx = hit_obj[i][0] - hit_obj[i-1][0]
        dy = hit_obj[i][1] - hit_obj[i-1][1]
        all_distances.append(math.sqrt(dx**2 + dy**2)) 
    
    mean_distance   = round(sum(all_distances) / len(all_distances), 2) if all_distances else 0.0
    std_distance    = round(math.sqrt(sum((d - mean_distance)**2 for d in all_distances) / len(all_distances)), 2) if all_distances else 0.0
    


    # --------------------------------------------------------------------------------------------------
    # intervalos de notas


    length_ms = hit_obj[-1][2] - hit_obj[0][2] if hit_obj else 0

    times = [t for _,_,t, _ in hit_obj]
    intervals = [times[i] - times[i-1] for i in range(1, len(times))]
    
    if intervals:
        mean_interval = sum(intervals) / len(intervals)
        var_interval = sum((x - mean_interval)**2 for x in intervals) / len(intervals)

    else:
        mean_interval = var_interval = 0.0

    stream_density = 0.0
    if base_bpm and intervals:
        stream_threshold = 60000 / (base_bpm * 4) * 1.1
        stream_count = sum(1 for iv in intervals if iv <= stream_threshold)
        stream_density = stream_count / len(intervals)

    circles = sum(1 for _,_,_, t in hit_obj if t & 1 and not (t & 2) and not (t & 8))
    sliders = sum(1 for _,_,_, t in hit_obj if t & 2)
    spinners = sum(1 for _,_,_, t in hit_obj if t & 8)
   

    # --------------------------------------------------------------------------------------------------
    # retorno dos dados:


    return {

        "circles":              circles,
        "sliders":              sliders,
        "spinners":             spinners,
        "total_notes":          total_notes,
        "mean_distance":        mean_distance,
        "std_distance":         std_distance,
        "base_bpm":             round(base_bpm,2) if base_bpm else None,
        "slider_ratio":         round(sliders / total_notes, 4) if total_notes else 0,
        "kiai_section_count":   len(kiai_ranges),
        "kiai_note_count":      len(kiai_obj),
        "kiai_note_ratio":      round(len(kiai_obj) / total_notes, 4) if total_notes else 0,
        "mean_kiai_dist":       round(mean_kiai_distance,2),
        "stream_density":       round(stream_density, 4),
        "interval_variance":    round(var_interval,2),
        "mean_interval_ms":     round(mean_interval),
        "notes_per_second":     round(total_notes / (length_ms / 1000), 4) if length_ms else 0,
        "mean_velocity":        round(mean_distance / mean_interval if mean_interval else 0.0, 4)
    }


# --------------------------------------------------------------------------------------------------
