import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np


# --------------------------------------------------------------------------------------------------
# distribuições por label:


FEATURES = ["ar", "cs", "od", "star_rating", "base_bpm"]


def plot_edas(df: pd.DataFrame, stats_df: pd.DataFrame):

    for feat in FEATURES:

        fig, axes = plt.subplots(3, 5, figsize=(20, 12), sharey=False)
        fig.suptitle(f"Distribution of '{feat}' per label", fontsize=14)

        for ax, (label, group) in zip(axes.flat, df.groupby("label")):
            values = group[feat].dropna()
        
            
            # --- clip outliers using IQR before plotting
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr     = q3 - q1
            clipped = values.clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

            # --------------------------------------------------------------------------------------------------
            # histogram + kde
            
            ax.hist(clipped, bins=20, density=True, alpha=0.5, color="steelblue", edgecolor="none")
            try:
                kde_x = np.linspace(clipped.min(), clipped.max(), 200)
                kde   = stats.gaussian_kde(clipped)
                ax.plot(kde_x, kde(kde_x), color="steelblue", linewidth=1.5)
            except Exception:
                pass
        
            
            # --------------------------------------------------------------------------------------------------
            # overlay normal curve for comparison

            x = np.linspace(clipped.min(), clipped.max(), 200)

            if clipped.std() > 0:
                ax.plot(x, stats.norm.pdf(x, clipped.mean(), clipped.std()),
                        color="red", linewidth=1, linestyle="--", alpha=0.7, label="Normal (referência)")

            ax.set_title(f"{label} (n={len(clipped)})", fontsize=8)
            ax.set_xlabel(feat, fontsize=7)
            ax.tick_params(labelsize=6)
    
        fig.text(0.01, 0.01,
                 "Azul: distribuição real dos dados (KDE)  |  Vermelho tracejado: curva normal de referência  |  Outliers limitados por IQR ±1.5",
                 fontsize=7, color="gray")


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
# confiabilidade das classes:


def reliability_check(df: pd.DataFrame, subjects: list[str], classes: list[str]):
    '''
    '''
    mask = df["label"].isin(subjects)
    subset = df[mask][classes]
    subset = subset.groupby("label").describe().T
    subset = subset.drop(["min", "max", "25%", "50%", "75%"], level=1)

    subset = subset.round(2)
    subset.to_csv("results/eda/analysis.csv", index=True)
    
    print("Analysis of class reliability saved on results/eda/analysis.csv")
    

# --------------------------------------------------------------------------------------------------
# mapa de correlação de pearson:


def correlation_map(df: pd.DataFrame, features: list[str], output: str = "results/eda/correlation.png"):
    '''
        Gera um heatmap de correlação de Pearson entre as features selecionadas.
        Valores próximos de +1 indicam correlação positiva forte,
        -1 correlação negativa forte, e 0 ausência de correlação linear.
    '''
    
    subset = df[features].dropna()
    corr = subset.corr(method="pearson")

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlação de Pearson entre features", fontsize=14)
    plt.tight_layout()
    plt.savefig(output, dpi=120)
    plt.close()

    print(f"Saved correlation map → {output}")


# --------------------------------------------------------------------------------------------------
# gráfico de barras horizontal — quantidade de mapas por label:


def plot_label_counts(df: pd.DataFrame, output: str = "results/eda/label_counts.png"):
    '''
        Plota um gráfico de barras horizontal mostrando quantos mapas existem
        por label, ordenado do maior para o menor.
    '''

    counts = df["label"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(counts) * 0.4)))
    counts.plot.barh(ax=ax, color="steelblue", edgecolor="none")

    ax.set_xlabel("Quantidade de mapas")
    ax.set_ylabel("Label")
    ax.set_title("Quantidade de mapas por label")

    for i, v in enumerate(counts):
        ax.text(v + max(counts) * 0.01, i, str(v), va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=120)
    plt.close()

    print(f"Saved label counts → {output}")


# --------------------------------------------------------------------------------------------------
