import numpy as np
import pandas as pd
from scipy import stats
from mod import to_dt, to_hr

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------------------------------------------------
# label stat analysis:


features = ["ar", "cs", "od", "drain", "difficulty_rating", "base_bpm"]

MOD_FEATURES = {
        "dt": ["base_bpm", "notes_per_second", "mean_interval_ms"],   # DT raises these
        "hr": ["ar", "cs", "od", "drain"],                             # HR raises these
    }
   

def stats_analysis(df: pd.DataFrame):
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
            4. Tabelar quantos desvios padrões isso contem;
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

                "label":    label,
                "feature":  feat,
                "mean":         round(mean, 4),
                "std":          round(std, 4),
                "variance":     round(var, 4),
                "skewness":     round(skew, 4),         # 0 = symmetric, >0 right tail, <0 left tail
                "kurtosis":     round(kurtosis, 4),     # 0 = normal, >0 heavy tails
                "p_normal":     round(p_normal, 4),     # >0.05 = likely normal distribution
                "is_normal":    p_normal > 0.05,
                "n":            len(values),

            })
    
    stats_df = pd.DataFrame(statistics)
    stats_df.to_csv("results/eda/eda_stats.csv", index=False)
    
    print("Saved stats.")
    print(stats_df.to_string())

    return stats_df
     

# --------------------------------------------------------------------------------------------------
# construíndo expectativas sobre cada feature de 
# engineering que deriva da mudança dos mods


def build_expectations(stats_df: pd.DataFrame, MOD_FEATURES):
    
    expectations = {}
    for feat in MOD_FEATURES:
        subs = stats_df[stats_df["feature"] == feat]

        expectations[feat] = {

            row["label"]: {"mean" : row["mean"], "std": row["std"]}
            for _, row in subs.iterrows() 
        }

    return expectations


# --------------------------------------------------------------------------------------------------
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
    SENSITIVITY = 1.5
     
    
    label = row["label"]
    flags = {"dt": 0, "hr": 0}
    
    if "dt" in label:
        return 0

    if "hr" in label:
        return 1

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


# --------------------------------------------------------------------------------------------------
# apply mods:


def apply_mods(df: pd.DataFrame, stats_df: pd.DataFrame):

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
    augmented_df.to_csv("data/processed/augmented.csv", index=False)

    print(f"Original rows: {len(df)}")
    print(f"Augmented rows: {len(augmented_rows)}")
    print(f"Total: {len(augmented_df)}")


# --------------------------------------------------------------------------------------------------
# vendo os dados visualmente:


def plot_edas(df: pd.DataFrame, stats_df: pd.DataFrame):

    for feat in features:

        fig, axes = plt.subplots(3, 5, figsize=(20, 12), sharey=False)
        fig.suptitle(f"Distribution of '{feat}' per label", fontsize=14)

        for ax, (label, group) in zip(axes.flat, df.groupby("label")):
            values = group[feat].dropna()
        
            
            # --------------------------------------------------------------------------------------------------
            # histogram + kde
            
            ax.hist(values, bins=20, density=True, alpha=0.5, color="steelblue", edgecolor="none")
            try:
                kde_x = np.linspace(values.min(), values.max(), 200)
                kde   = stats.gaussian_kde(values)
                ax.plot(kde_x, kde(kde_x), color="steelblue", linewidth=1.5)
            except Exception:
                pass
        
            
            # --------------------------------------------------------------------------------------------------
            # overlay normal curve for comparison

            x = np.linspace(values.min(), values.max(), 200)
            ax.plot(x, stats.norm.pdf(x, values.mean(), values.std()),
                    color="red", linewidth=1, linestyle="--", alpha=0.7)

            ax.set_title(f"{label} (n={len(values)})", fontsize=8)
            ax.set_xlabel(feat, fontsize=7)
            ax.tick_params(labelsize=6)

        # hide unused subplots if labels < 15
        for ax in axes.flat[df["label"].nunique():]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig(f"results/eda/dist_{feat}.png", dpi=120)
        plt.close()

        print(f"Saved dist_{feat}.png")
        
        
    # --------------------------------------------------------------------------------------------------

    p_pivot = stats_df.pivot(index="label", columns="feature", values="p_normal")

    plt.figure(figsize=(10, 8))
    sns.heatmap(p_pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=0.1)
    plt.title("Shapiro-Wilk p-value per label × feature\ngreen > 0.05 = normal | red = non-normal distribution")
    plt.tight_layout()
    plt.savefig("results/eda/normality_heatmap.png", dpi=120)
    plt.close()

    print("Saved normality_heatmap.png")


# --------------------------------------------------------------------------------------------------
# teste se necessário:


if __name__ == "__main__":
        

    df = pd.read_csv("data/processed/dataset.csv")
    
    df["label"] = df["label"].str.replace("farm", "dt_farm")

    # rodar análise
    stats_df = stats_analysis(df)

    # rodar EDAs
    plot_edas(df, stats_df)
    
    # rodar possível aplicação de mods
    apply_mods(df, stats_df)
    

# --------------------------------------------------------------------------------------------------
