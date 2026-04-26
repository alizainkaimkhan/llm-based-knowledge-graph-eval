import pandas as pd
import random
from collections import defaultdict

def run_relation_aware_corruption(name, sampled_csv, full_txt, sep='\t'):
    print(f"\n{'='*25} {name} Triplets Corruption {'='*25}")

    # Load Data
    df_sampled = pd.read_csv(sampled_csv)
    df_full = pd.read_csv(full_txt, sep=sep, header=None, names=['head', 'relation', 'tail'])

    # Global Truth Set for the Filter Rule
    truth_set = set(zip(df_full['head'], df_full['relation'], df_full['tail']))

    # Build Relation-Aware Candidate Pools (Domain/Range)
    # We identify every entity that has ever acted as a 'head' or 'tail' for each relation
    head_pools = defaultdict(set)
    tail_pools = defaultdict(set)
    all_relations = list(df_full['relation'].unique())
    all_entities = list(set(df_full['head']) | set(df_full['tail']))

    for _, row in df_full.iterrows():
        h, r, t = row['head'], row['relation'], row['tail']
        head_pools[r].add(h)
        tail_pools[r].add(t)

    print(f"Knowledge Base Loaded. Total Entities: {len(all_entities)}, Relations: {len(all_relations)}")

    final_rows = []

    # Generation Loop
    for i, row in df_sampled.iterrows():
        h, r, t = row['head'], row['relation'], row['tail']

        # Add the Positive Fact (Label 1)
        final_rows.append({
            'head': h, 'relation': r, 'tail': t,
            'label': 1, 'corruption_type': 'none'
        })

        # Determine corruption mode (50/50 balance between Entity and Relation corruption)
        mode = 'entity' if i % 2 == 0 else 'relation'

        while True:
            if mode == 'entity':
                # 50/50 chance to corrupt Head or Tail
                corrupt_head = random.choice([True, False])

                if corrupt_head:
                    # Pick from entities that have appeared in the Head position for this relation
                    pool = list(head_pools[r])
                    # Safety check: if pool is only the current entity, fallback to global
                    if len(pool) <= 1: pool = all_entities
                    new_ent = random.choice(pool)
                    candidate = (new_ent, r, t)
                    c_tag = 'head_corrupted'
                else:
                    # Pick from entities that have appeared in the Tail position for this relation
                    pool = list(tail_pools[r])
                    if len(pool) <= 1: pool = all_entities
                    new_ent = random.choice(pool)
                    candidate = (h, r, new_ent)
                    c_tag = 'tail_corrupted'
            else:
                # Relation Corruption: Swap the relation ID
                new_rel = random.choice(all_relations)
                candidate = (h, new_rel, t)
                c_tag = 'relation_corrupted'

            # Filter Rule: Must not be a known truth AND must be different from original
            if candidate not in truth_set and candidate != (h, r, t):
                final_rows.append({
                    'head': candidate[0], 'relation': candidate[1], 'tail': candidate[2],
                    'label': 0, 'corruption_type': c_tag
                })
                break

    # 4. Combine and Shuffle
    final_df = pd.DataFrame(final_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    output_name = f"data/{name}_test_numeric_1000.csv"
    final_df.to_csv(output_name, index=False)

    print(f"Success! Created {output_name} ({len(final_df)} rows)")
    print("Breakdown by Corruption Type:")
    print(final_df['corruption_type'].value_counts())
    return final_df

if __name__ == "__main__":
    run_relation_aware_corruption("FB15k-237", "data/FB15k-237_valid_numeric_500.csv", "data/fb15k_test_id.tsv")
    run_relation_aware_corruption("WN18RR", "data/WN18RR_valid_numeric_500.csv", "data/wn_test_id.txt")