import pandas as pd
import os

# Define input files configuration
DATASETS = {
    "FB15k-237": {"file": "data/fb15k_test_id.tsv", "sep": "\t"},
    "WN18RR": {"file": "data/wn_test_id.txt", "sep": "\t"}
}

# Generate statistical summary
def get_summary(df, label, sort_order=None):
    counts = df['relation'].value_counts()
    percentages = (df['relation'].value_counts(normalize=True) * 100).round(2)
    summary = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages})
    summary.index.name = 'Relation ID'
    if sort_order is not None:
        summary = summary.reindex(sort_order)
    print(f"\n--- {label} ---")
    print(summary.head(10))
    return summary.index.tolist()

def prepare_data():
    for name, config in DATASETS.items():
        print(f"\n{'='*25} {name} (Sampling Summary) {'='*25}")
        if not os.path.exists(config['file']):
            print(f"Error: {config['file']} not found.")
            continue

        # Load triplets
        df = pd.read_csv(config['file'], sep=config['sep'], header=None, names=['head', 'relation', 'tail'])
        print(f"Original Triplet Count: {len(df)}")
        original_order = get_summary(df, f"Original {name} Summary")

        # Stratified Sampling (Target 500)
        target_n = 500
        rel_counts = df['relation'].value_counts(normalize=True)
        samples = []
        for rel, proportion in rel_counts.items():
            rel_subset = df[df['relation'] == rel]
            n_to_sample = max(1, int(round(proportion * target_n)))
            samples.append(rel_subset.sample(n=min(len(rel_subset), n_to_sample), random_state=42))

        valid_triplets = pd.concat(samples).sample(frac=1, random_state=42).head(target_n)
        get_summary(valid_triplets, f"Sampled {name} Summary", sort_order=original_order)

        output_name = f"data/{name}_valid_numeric_500.csv"
        valid_triplets.to_csv(output_name, index=False)
        print(f"\nSaved to {output_name}")

if __name__ == "__main__":
    prepare_data()