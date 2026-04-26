import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load Datasets
llama_fb = pd.read_csv("results/results_llama_FB15k-237_mapped_final.csv")
gpt_fb = pd.read_csv("results/results_gpt_FB15k-237_mapped_final.csv")
llama_wn = pd.read_csv("results/results_llama_WN18RR_mapped_final.csv")
gpt_wn = pd.read_csv("results/results_gpt_WN18RR_mapped_final.csv")

def get_metrics(df, model_name, dataset_name):
    rows = []
    prefix = "llama" if "Llama" in model_name else "gpt"
    for s in ["S1", "S2", "S3", "S4"]:
        col = f"{prefix}_{s}_pred"
        if col in df.columns:
            valid = df[df[col] != -1]
            rows.append({
                "Dataset": dataset_name, "Model": model_name, "Strategy": s,
                "Accuracy": accuracy_score(valid['label'], valid[col]),
                "Precision": precision_score(valid['label'], valid[col], zero_division=0),
                "Recall": recall_score(valid['label'], valid[col], zero_division=0),
                "F1": f1_score(valid['label'], valid[col], zero_division=0)
            })
    return pd.DataFrame(rows)

# Generate Master Metrics Table
metrics = pd.concat([
    get_metrics(llama_fb, "Llama-3-70B", "FB15k-237"),
    get_metrics(gpt_fb, "GPT-5.4 mini", "FB15k-237"),
    get_metrics(llama_wn, "Llama-3-70B", "WN18RR"),
    get_metrics(gpt_wn, "GPT-5.4 mini", "WN18RR")
])

# --- VISUALIZATION 1: Grouped Bar Chart ---
plt.figure(figsize=(10, 6))
# Change your barplot line to this:
sns.barplot(data=metrics, x="Strategy", y="F1", hue="Model", palette="coolwarm", errorbar=None)
plt.title("F1-Score Comparison: Llama vs GPT across Strategies")
plt.savefig("results/hero_f1_grouped_bar.png")

# --- VISUALIZATION 2: F1-SCORE HEATMAP ---
plt.figure(figsize=(10, 6))
heatmap_data = metrics.pivot_table(index=['Dataset', 'Model'], columns='Strategy', values='F1')
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title("F1-Score Heatmap: Model/Dataset vs Strategy")
plt.tight_layout()
plt.savefig("results/f1_heatmap.png")

# --- VISUALIZATION 3: CONFUSION MATRICES (S4) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
cms = [(llama_fb, "llama", "FB15k-237", axes[0,0]), (gpt_fb, "gpt", "FB15k-237", axes[0,1]),
       (llama_wn, "llama", "WN18RR", axes[1,0]), (gpt_wn, "gpt", "WN18RR", axes[1,1])]
for df, prefix, ds, ax in cms:
    col = f"{prefix}_S4_pred"
    cm = confusion_matrix(df['label'], df[col])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_title(f"CM: {prefix.upper()} on {ds} (S4)")
plt.tight_layout()
plt.savefig("results/confusion_matrices_s4.png")

# --- VISUALIZATION 4: RADAR CHARTS (Corruption Accuracy) ---
def plot_radar(ds_name):
    corruption_types = ['head_corrupted', 'tail_corrupted', 'relation_corrupted']
    labels = np.array(['Head', 'Tail', 'Rel'])

    def get_accs(df, prefix):
        return [accuracy_score(df[df['corruption_type']==ct]['label'],
                               df[df['corruption_type']==ct][f"{prefix}_S4_pred"]) for ct in corruption_types]

    l_accs = get_accs(llama_fb if ds_name=="FB15k-237" else llama_wn, "llama")
    g_accs = get_accs(gpt_fb if ds_name=="FB15k-237" else gpt_wn, "gpt")

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    l_accs += l_accs[:1]; g_accs += g_accs[:1]; angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, l_accs, color='blue', label='Llama-3-70B')
    ax.fill(angles, l_accs, color='blue', alpha=0.25)
    ax.plot(angles, g_accs, color='red', label='GPT-5.4 mini')
    ax.fill(angles, g_accs, color='red', alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.title(f"Corruption Detection Accuracy (S4) - {ds_name}")
    plt.legend(loc='lower right')
    plt.savefig(f"results/radar_corruption_{ds_name}.png")

plot_radar("FB15k-237")
plot_radar("WN18RR")

print("All plots generated successfully!")