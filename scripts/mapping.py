import pandas as pd

def perform_final_mapping():
    print(f"{'='*25} Phase 4: Natural Language Mapping {'='*25}")

    # FB15k-237 mapping logic
    print("\nMapping FB15k-237...")
    fb_test = pd.read_csv('data/FB15k-237_test_numeric_1000.csv')

    # Load the three mapping files for FB15k
    fb_ent_id2mid = pd.read_csv('data/fb15k_entity_id_to_text.tsv', sep='\t', header=None, names=['numeric_id', 'mid'])
    fb_mid2name = pd.read_csv('data/FB15k_mid2name.txt', sep='\t', header=None, names=['mid', 'name'])
    fb_rel_id2text = pd.read_csv('data/fb15k_relation_id_to_text.tsv', sep='\t', header=None, names=['numeric_id', 'relation_text'])

    # Join Entity ID -> MID -> English Name
    fb_ent_map_df = pd.merge(fb_ent_id2mid, fb_mid2name, on='mid')
    fb_id2name = dict(zip(fb_ent_map_df['numeric_id'], fb_ent_map_df['name']))
    fb_id2rel = dict(zip(fb_rel_id2text['numeric_id'], fb_rel_id2text['relation_text']))

    def clean_fb_rel(rel_path):
        # Cleans "/people/person/nationality" into "people person nationality"
        return rel_path.replace('/', ' ').replace('_', ' ').strip()

    fb_test['head_name'] = fb_test['head'].map(lambda x: fb_id2name.get(x, f"ID_{x}"))
    fb_test['tail_name'] = fb_test['tail'].map(lambda x: fb_id2name.get(x, f"ID_{x}"))
    fb_test['relation_name'] = fb_test['relation'].map(lambda x: clean_fb_rel(fb_id2rel.get(x, str(x))))

    # Create the text prompt string
    fb_test['natural_language_triplet'] = fb_test['head_name'] + " [" + fb_test['relation_name'] + "] " + fb_test['tail_name']
    fb_test.to_csv('data/FB15k-237_mapped_final.csv', index=False)
    print("FB15k-237 Mapping Complete.")

    # WN18RR mapping logic
    print("\nMapping WN18RR...")
    wn_test = pd.read_csv('WN18RR_test_numeric_1000.csv')

    # Map IDs to Text by matching lines in the original source files
    wn_id2text = {}
    with open('data/wn_test_id.txt', 'r') as f_id, open('data/wn_test_text.txt', 'r') as f_text:
        for line_id, line_text in zip(f_id, f_text):
            ids = line_id.strip().split('\t')
            texts = line_text.strip().split('\t')
            if len(ids) == 3 and len(texts) == 3:
                # Store mappings for both heads and tails
                wn_id2text[ids[0]] = texts[0]
                wn_id2text[ids[2]] = texts[2]

    def clean_wn_rel(rel):
        return rel.replace('_', ' ').strip()

    # Apply padding to IDs to ensure they match the 8-digit format
    wn_test['head_name'] = wn_test['head'].astype(str).str.zfill(8).map(lambda x: wn_id2text.get(x, f"ID_{x}"))
    wn_test['tail_name'] = wn_test['tail'].astype(str).str.zfill(8).map(lambda x: wn_id2text.get(x, f"ID_{x}"))
    wn_test['relation_name'] = wn_test['relation'].apply(clean_wn_rel)

    wn_test['natural_language_triplet'] = wn_test['head_name'] + " [" + wn_test['relation_name'] + "] " + wn_test['tail_name']
    wn_test.to_csv('data/WN18RR_mapped_final.csv', index=False)
    print("WN18RR Mapping Complete.")

    # Mapping verification
    print("\n--- Mapping Verification Samples ---")
    print("\nFB15k-237 Sample:")
    print(fb_test[['natural_language_triplet', 'label', 'corruption_type']].head(3))
    print("\nWN18RR Sample:")
    print(wn_test[['natural_language_triplet', 'label', 'corruption_type']].head(3))

if __name__ == "__main__":
    perform_final_mapping()