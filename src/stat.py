from src.mod import to_dt, to_hr

from scipy import stats
import pandas as pd



# ──────────────────────────────────────────────────────────────────────────────────────────────────
# label stat analysis:


features = ["ar", "cs", "od", "star_rating", "base_bpm"]

MOD_FEATURES =  {   "dt": ["base_bpm", "notes_per_second", "mean_interval_ms"], # DT raises these
                    "hr": ["ar", "cs", "od"],                                   # HR raises these
                }
   

def stats_analysis(df: pd.DataFrame) -> pd.DataFrame:
    '''
        
        • Aqui, vamos analisar estatisticamente as features base seguintes:
                
            ◦ Advindos do .osu:
                → HPDrainRate          
                → CircleSize            
                → OverallDificulty
                → ApproachRate

            ◦ Advindos dos .json's:
                → difficulty_rating
            
            ❓: a questão do BPM: pegar via beat_length ou json?

            
        Quais serão as métricas estatísticas utilizadas aqui?
            
            1. Obtenção da média para todas as features base,
            2. Obtenção do desvio padrão para todas as features base,
            3. Valor Médio / Valor Base: análise desse coeficiente resultado
            4. Tabelar quantos desvios padrões isso contêm

    ''' 

    df["label"] = df["label"].str.replace("_maps", "")
    df = df.dropna(subset=features)

    statistics = [] 
        
    for label, group in df.groupby("label"):
        for feat in features:

            values = group[feat].dropna()
            mean = values.mean()
            std = values.std()
            var = values.var()

            skew = values.skew()
            kurtosis = values.kurtosis()

            
            sample = values.sample(min(len(values), 500), random_state=42)
            _, p_normal = stats.shapiro(sample) 
    
            statistics.append({

                "label":        label,
                "feature":      feat,
                "mean":         round(mean, 2),
                "std":          round(std, 2),
                "variance":     round(var, 2),
                "skewness":     round(skew, 2),         # 0 = symmetric, >0 right tail, <0 left tail
                "kurtosis":     round(kurtosis, 2),     # 0 = normal, >0 heavy tails
                "p_normal":     round(p_normal, 2),     # >0.05 = likely normal distribution
                "is_normal":    p_normal > 0.05,
                "n":            len(values),

            })
    
    stats_df = pd.DataFrame(statistics)
    stats_df.to_csv("results/eda/eda_stats.csv", index=False)
    
    print("\nSaved statistics.")
    #print(stats_df.to_string())


    return stats_df
     

# ──────────────────────────────────────────────────────────────────────────────────────────────────
# construíndo expectativas sobre cada feature de 
# engineering que deriva da mudança dos mods


def build_expectations(stats_df: pd.DataFrame, MOD_FEATURES):
    '''
        Constrói um dicionário de "expectativas" estatísticas para cada feature
        que é afetada por mods.

        Para cada feature da lista MOD_FEATURES, extraímos a média e o desvio
        padrão por label. Isso permite comparar, depois, se um mapa individual
        está "fora do esperado" para a sua label — e, portanto, se provavelmente
        precisa de um mod para se adequar à distribuição esperada.

        Retorno:
            {
                "base_bpm": {
                    "stream":  {"mean": 180, "std": 25},
                    "jump":    {"mean": 200, "std": 30},
                    ...
                },
                "ar": { ... },
                ...
            }
    '''
    
    expectations = {}
    for feat in MOD_FEATURES:
        subs = stats_df[stats_df["feature"] == feat]

        expectations[feat] = {

            row["label"]: {"mean" : row["mean"], "std": row["std"]}
            for _, row in subs.iterrows() 
        }


    return expectations


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# vendo se são necessários mods no mapa:


def are_mods_needed(row, expectations):
    '''
       
        Os critérios nessa função são os seguintes:

            1. quando vem, pelo NOME da label do mapa, a informação de {mod}...map            
            2. quando, pela lógica da label em relação ao mapa, há necessidade de mod 
                
                → ex: label=speed, mapa com bpm=150
            
            3. quando, depois de análise estatística feita nos mapas de uma label singular,
               for observado que um mapa possui variação considerável em 2+ features.
            

        No final, usaremos, POR ORA:

            0: DT
            1: HR
        
            ↓ WIP
            
            2: HD
            3: EZ
            4: FL
    '''
    
    # infos que precisamos usar de target
    MIN_FLAGS = 2


    '''
    Limiar de sensibilidade em desvios padrão.
    
    Se o valor de uma feature do mapa está com LABEL_SENSITIVITY:

        > +SENSITIVITY  →  mapa está MUITO ACIMA da média da label (mais difícil que o esperado)
        < -SENSITIVITY  →  mapa está MUITO ABAIXO da média da label (mais fácil que o esperado), precisa de mod.
    
    '''
    SENSITIVITY = 1.5
     

    label = row["label"]
    flags = {"dt": 0, "hr": 0}
    
    # casos de obviedade, ex: dt_farm obviamente precisa de transformação dt.
    if "dt" in label:
        return 0

    if "hr" in label:
        return 1
    
    
    #if "precision" in label:
    #    return 1


    for mod, feats in MOD_FEATURES.items():
        for feat in feats:

            if feat not in expectations or label not in expectations[feat]:
                continue

            mean = expectations[feat][label]["mean"]
            std = expectations[feat][label]["std"]
            
            if std == 0: 
                continue
            
            LABEL_SENSITIVITY = (row[feat] - mean) / std
            
            if LABEL_SENSITIVITY < -SENSITIVITY:
                flags[mod] += 1 


    dt = flags["dt"] >= MIN_FLAGS
    hr = flags["hr"] >= MIN_FLAGS
    
    if dt: return 0
    if hr: return 1
    

    return None


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# apply mods:


def apply_mods(df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:

    expectations = build_expectations(stats_df, MOD_FEATURES["dt"] + MOD_FEATURES["hr"]) 
    
    augmented_rows = []

    for _, row in df.iterrows():

        mod = are_mods_needed(row, expectations)
        if mod is None:
            continue
    
        new_row = row.copy()

        match mod:
            case 0: # dt
                new_row = to_dt(row, new_row)
 
            case 1: # hr
                new_row = to_hr(row, new_row)
            
        new_row["mod"] = mod
        augmented_rows.append(new_row)

    augmented_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    #augmented_df.to_csv("data/processed/augmented.csv", index=False)
    
    print("Dataset updated:")
    print(f"  Original rows     : {len(df)}")
    print(f"  Augmented rows    : {len(augmented_rows)}")
    print(f"  New total         : {len(augmented_df)}")

    return augmented_df


# ──────────────────────────────────────────────────────────────────────────────────────────────────
