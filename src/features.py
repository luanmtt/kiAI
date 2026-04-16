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
            

        if kiai_active and kiai_on is None:
            kiai_on = time

        elif not kiai_active and kiai_on is not None:
            kiai_ranges.append((kiai_on, time))
            kiai_on = None


    if kiai_on is not None:
        kiai_ranges.append((kiai_on, float("inf")))


    base_bpm = round(base_bpm,2) if base_bpm else None



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

    
    
    # --------------------------------------------------------------------------------------------------
    # função auxiliar

    def in_kiai(t):
        return any(start <= t <= end for start, end in kiai_ranges)
    

    # --------------------------------------------------------------------------------------------------
    # distâncias entre objetos enquanto o kiai é ativo.


    
    kiai_obj = [(x,y,t) for x,y,t, _ in hit_obj if in_kiai(t)]
    
    # 
    kiai_distances = []
    for i in range(1, len(kiai_obj)):

        k_dx = kiai_obj[i][0] - kiai_obj[i-1][0]
        k_dy = kiai_obj[i][1] - kiai_obj[i-1][1]
        kiai_distances.append(math.sqrt(k_dx**2 + k_dy**2)) 


    mean_kiai_distance = (
        sum(kiai_distances) / len(kiai_distances) if kiai_distances else 0.0
    )
    
    
    # tamanho, em notas, do kiai.
    # nos ajuda a determinar categorias com muitas notas/kiai : stream, tech.
    kiai_note_ratio = round(len(kiai_obj) / total_notes, 4) if total_notes else 0
    
    # distância média dentro do kiai.
    # nos ajuda a distinguir mapas com baixa / altas distâncias.
    kiai_mean_dist  = round(mean_kiai_distance,2)
    
    kiai_section_count = len(kiai_ranges)
    kiai_note_count = len(kiai_obj)



    # --------------------------------------------------------------------------------------------------
    # análise estatística de todas as distâncias do mapa:
    


    # lista coletada para computar as features de distância
    all_distances = []
    for i in range(1, len(hit_obj)):
        dx = hit_obj[i][0] - hit_obj[i-1][0]
        dy = hit_obj[i][1] - hit_obj[i-1][1]

        all_distances.append(math.sqrt(dx**2 + dy**2)) 
    

    # desenvolvimento estatístico das DISTÂNCIAS ESPACIAIS entre as notas
    mean_distance   = round(sum(all_distances) / len(all_distances), 2) if all_distances else 0.0
    std_distance    = round(math.sqrt(sum((d - mean_distance)**2 for d in all_distances) / len(all_distances)), 2) if all_distances else 0.0


    # coleção de dados sobre o tipo de hit_obj
    circles = sum(1 for _,_,_, t in hit_obj if t & 1 and not (t & 2) and not (t & 8))
    sliders = sum(1 for _,_,_, t in hit_obj if t & 2)
    spinners = sum(1 for _,_,_, t in hit_obj if t & 8)
    


    # --------------------------------------------------------------------------------------------------
    # intervalos de notas



    # variáveis coletadas para desenvolver as features de intervalos
    length_ms = hit_obj[-1][2] - hit_obj[0][2] if hit_obj else 0
    times = [t for _,_,t, _ in hit_obj]
    intervals = [times[i] - times[i-1] for i in range(1, len(times))]
    

    if intervals:
        # desenvolvimento estatístico dos INTERVALOS TEMPORAIS entre as notas.
        mean_interval   = round(sum(intervals) / len(intervals))
        var_interval    = round(sum((x - mean_interval)**2 for x in intervals) / len(intervals), 2)
        std_interval    = round(math.sqrt(sum((x - mean_interval)**2 for x in intervals) / len(intervals)), 2) if intervals else 0.0

    else:
        mean_interval = var_interval = std_interval = 0.0
    


    stream_density = 0.0
    if base_bpm and intervals:

        stream_threshold = 60000 / (base_bpm * 4) * 1.1
        stream_count = sum(1 for iv in intervals if iv <= stream_threshold)
       

        # proporção entre tempo_stream : tempo_total
        # muito bom para determinar stream / stamina
        stream_density = round(stream_count / len(intervals), 4)

    
    # a proporção de sliders : notas totais.
    # pode ser útil na identificação de tech, consistency
    slider_ratio = round(sliders / total_notes, 4) if total_notes else 0
    

    # velocidade média do mapa.
    # pode ser útil na separação de jump / stream 
    mean_velocity = round(mean_distance / mean_interval if mean_interval else 0.0, 4)
   

    # notas por segundo.
    # muito útil para identificar stamina / stream
    notes_per_second = round(total_notes / (length_ms / 1000), 4) if length_ms else 0.0



    # --------------------------------------------------------------------------------------------------
    # misc features para distinção de 'tech', 'consistency'



    # complexidade de ritmo:    → alto = muitas mudanças de ritmo, indicativo de tech, gimmick
    #                           → baixo = poucas mudanças, stream, stamina
    unique_variations = len(set(round(ivs, 0) for ivs in intervals))
    rhythm_complexity = round(unique_variations / len(intervals), 4) if intervals else 0.0
    

    # coeficiente de variação dos intervalos:   → alto: alt, tech, gimmick
    #                                           → baixo: stream, stamina
    interval_cv = round((std_interval / mean_interval), 4) if mean_interval else 0.0
    


    # --------------------------------------------------------------------------------------------------
    # retorno dos dados:


    return {

        "circles":              circles,
        "sliders":              sliders,
        "spinners":             spinners,
        "slider_ratio":         slider_ratio,
        "base_bpm":             base_bpm,

        "total_notes":          total_notes,
        "mean_distance":        mean_distance,
        "std_distance":         std_distance,
        "stream_density":       stream_density,
        
        #"kiai_section_count":   kiai_section_count,
        "kiai_note_ratio":      kiai_note_ratio,
        "kiai_mean_dist":       kiai_mean_dist,
        "kiai_note_count":      kiai_note_count,
                   
        "interval_variance":    var_interval,
        "interval_cv":          interval_cv,

        "mean_interval_ms":     mean_interval,
        "notes_per_second":     notes_per_second,
        "mean_velocity":        mean_velocity,
        
        "rhythm_complexity":    rhythm_complexity,

    }


# --------------------------------------------------------------------------------------------------
