import numpy as np


# --------------------------------------------------------------------------------------------------
'''

    A aplicação de mods no osu! modifica informações vitais de dados do mapa.
        
        → BPM   → CS
        → AR    → Drain
        → OD    → Total Length
    
    A mudança dessas infos resulta na mudança das seguintes features advindas de features.py
        
        → base_bpm          → mean_velocity
        → stream_density    → notes_per_second


    No escopo desse treino, DT e HR são must. EZ e Flashlight são mods de nicho, logo não serão
    abordados por ora. Entretanto, HD é um mapa presente na gameplay de qualquer jogador, e a sua
    ação nos dados do mapa é nula. Logo, há de explorar aonde a mudança do HD se demonstrará nos 
    dados.

    
    • Fórmulas Gerais retiradas da osu!wiki que talvez sejam necessárias:

        CS:     r = (54.4 - 4.48 * CS) * 1.00041
        
        OD:     score   hit-window(ms)
                _______________________
                300	    80 - 6 × OD
                100	    140 - 8 × OD
                50	    200 - 10 × OD
        
        AR: 
                AR < 5: preempt = 1200ms + 120ms * (5 - AR)
                AR = 5: preempt = 1200ms
                AR > 5: preempt = 1200ms - 150ms * (AR - 5)

'''


# --------------------------------------------------------------------------------------------------
# cálculos de ar p/ dt

def ar_to_ms(ar):
    if ar < 5:   return 1800 - 120 * ar
    elif ar == 5: return 1200
    else:         return 1200 - 150 * (ar - 5)


def ms_to_ar(ms):
    if ms > 1200:  return (1800 - ms) / 120
    elif ms == 1200: return 5
    else:           return 5 + (1200 - ms) / 150


# DT perceived AR:
def perceived_ar_dt(ar):
    ms = ar_to_ms(ar)
    return round(ms_to_ar(ms / 1.5), 2)

def perceived_od_dt(od):
    window    = 80 - 6 * od
    window_dt = window / 1.5
    return round((80 - window_dt) / 6, 2)

# --------------------------------------------------------------------------------------------------


def to_dt(row, new_row):
    '''
        DT: 1.5x BPM speedup, logo redução de total_length em 33%.

        → AR base = 5
        → AR máxima = 11
        
        → Passamos na ordem [ar, total_length]

        ◦ Info Adicional:

            "The OD value is not affected, but due to the 50% play speed increase, hit windows are 33% shorter.

            The AR value is not affected, but due to the 50% play speed increase, hit 
            objects stay on screen for 33% less time.

            While Double Time do not change the AR value, the speed difference 
            leads to an apparent AR change. HT/DT ARs are commonly referred to in terms of their 
            perceived value. For example, "AR 8 + DT" may also be written as "AR 9.6"."
    '''


    new_row["base_bpm"]         = np.floor(row["base_bpm"] * 1.5)
    new_row["bpm"]              = np.floor(row["bpm"] * 1.5)
    new_row["total_length"]     = np.floor(row["total_length"] * 0.67)
    new_row["mean_interval_ms"] = row["mean_interval_ms"] / 1.5
    new_row["notes_per_second"] = row["notes_per_second"] * 1.5
    new_row["mean_velocity"]    = row["mean_velocity"] * 1.5
    new_row["ar"]               = perceived_ar_dt(row["ar"])
    new_row["od"]               = perceived_od_dt(row["od"])
    new_row["ar_od_ratio"]      = round(new_row["ar"] / new_row["od"], 4) if new_row["od"] else 0.0

    return new_row
    

# --------------------------------------------------------------------------------------------------


def to_hr(row, new_row):
    '''
        Aumenta CS em 1.3x, AR em 1.4x, Drain em 1.4x e OD em 1.4x
        
        → AR máxima = 10
        → OD máxima = 10
        → CS máximo = 10
        → HP máximo (drain) = 10

        → Passamos na ordem [cs, ar, drain, od]
    '''

    new_row["cs"]    = min(row["cs"]    * 1.3, 10)
    new_row["ar"]    = min(row["ar"]    * 1.4, 10)
    #new_row["drain"] = min(row["drain"] * 1.4, 10)
    new_row["od"]    = min(row["od"]    * 1.4, 10)

    return new_row

# --------------------------------------------------------------------------------------------------


#WIP
# def to_hd():


# --------------------------------------------------------------------------------------------------
