import math
import os

# --------------------------------------------------------------------------------------------------
# parsing file:

def parse_osu_file(filepath:str) -> dict:
    
    # lendo linha-a-linha cada .osu.

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    sections = {}
    current = None

    for line in content.splitlines():

        line = line.strip()
        if not line or line.startswith("//"):
            continue

        if line.startswith("[") and line.endswith("]"):

            current = line[1:-1]
            sections[current] = []
        
        elif current:
            sections[current].append(line)

    

    # --------------------------------------------------------------------------------------------------
    # pegando os valores mais internos aos mapas:
  

    # -------------------
    # kiai info:

    kiai_ranges = []    # aqui, temos como targets: start_ms, end_ms
    base_bpm = None
    kiai_on = None

    for line in sections.get("TimingPoints", []):

        parts = line.split(",")

        if len(parts) < 8:
            continue

        time        = float(parts[0])
        beat_length = float(parts[1])
        uninherited = int(parts[6])
        effects     = int(parts[7])
        kiai_active = bool(effects & 1)

        if uninherited == 1 and beat_length > 0:

            bpm = 60000 / beat_length

            if base_bpm is None:
                base_bpm = bpm # pegando o bpm dominante com essa lógica

            if kiai_active and kiai_on is None:
                kiai_on = time

            elif not kiai_active and kiai_on is not None:
                kiai_ranges.append((kiai_on, time))
                kiai_on = None

        if kiai_on is not None:
            kiai_ranges.append((kiai_on, float("inf")))
    

    # -------------------
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
    


    # -------------------
    # distâncias entre objetos enquanto o kiai é ativo.

    kiai_distances = []
    for i in range(1, len(kiai_obj)):

        dx = kiai_obj[i][0] - kiai_obj[i-1][0]
        dy = kiai_obj[i][1] - kiai_obj[i-1][1]
        kiai_distances.append(math.sqrt(dx**2 + dy**2)) 


    mean_kiai_distance = (
        sum(kiai_distances) / len(kiai_distances) if kiai_distances else 0.0
    )


    
    # -----------------
    # intervalos de notas

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

    circles = sum(1 for _,_,_, t in hit_obj if t & 1)
    sliders = sum(1 for _,_,_, t in hit_obj if t & 2)
    spinners = sum(1 for _,_,_, t in hit_obj if t & 3)
   

# --------------------------------------------------------------------------------------------------
# retorno dos dados:


    return {

        "base_bpm": round(base_bpm,2) if base_bpm else None,
        "kiai_section_count": len(kiai_ranges),
        "kiai_note_count": len(kiai_obj),
        "kiai_note_ratio": round(len(kiai_obj) / total_notes, 4) if total_notes else 0,
        "mean_kiai_dist": round(mean_kiai_distance,2),
        "interval_variance": round(var_interval,2),
        "mean_interval_ms": round(mean_interval),
        "stream_density": round(stream_density, 4),
        "circles": circles,
        "sliders": sliders,
        "spinners": spinners,
        "total_notes": total_notes,
    }



if __name__ == "__main__":
    # quick test on a single file
    import json
    sample = next(
        os.path.join("data/beatmaps", f)
        for f in os.listdir("data/beatmaps")
        if f.endswith(".osu")
    )
    print(f"Parsing: {sample}")
    result = parse_osu_file(sample)
    print(json.dumps(result, indent=2))
