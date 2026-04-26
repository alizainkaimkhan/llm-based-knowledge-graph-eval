import pandas as pd
from openai import OpenAI
import os
import time

# Configuration
client = OpenAI()

DATASETS = ["data/FB15k-237_mapped_final.csv", "data/WN18RR_mapped_final.csv"]
STRATEGIES = ["S1", "S2", "S3", "S4"]
MODEL = "gpt-5.4-mini"

# Define prompting strategies
def get_prompt(triplet, strategy):
    if strategy == "S1":
        return f"Is the following factual triplet true? Answer only with 'Yes' or 'No'.\nTriplet: {triplet}"
    elif strategy == "S2":
        return (f"Evaluate the factual accuracy of the final triplet based on the patterns shown in the examples. "
                f"Answer only with 'Yes' if true or 'No' if false.\n\n"
                f"Example 1: Paris [location contains] Eiffel Tower -> Yes\n"
                f"Example 2: Albert Einstein [born in] Tokyo -> No\n"
                f"Example 3: canine.n.01 [hypernym] dog.n.01 -> Yes\n"
                f"Example 4: apple.n.01 [has part] engine.n.01 -> No\n\n"
                f"Triplet: {triplet} ->")
    elif strategy == "S3":
        return (f"Examine the following triplet and explain your reasoning step-by-step before concluding "
                f"if it is true or false. Answer with 'True' or 'False' at the end.\nTriplet: {triplet}")
    elif strategy == "S4":
        return (f"You are a senior Knowledge Graph Engineer specializing in Freebase (real-world facts) "
                f"and WordNet (lexical semantics). Your task is to validate the following triple for factual integrity.\n\n"
                f"Instructions:\n1. Analyze the relationship between the head and tail entities.\n"
                f"2. Determine if this specific connection is recognized in a standard knowledge base.\n"
                f"3. Output exactly one word: 'True' or 'False'.\n\n"
                f"Triplet: {triplet}\nOutput:")
    return triplet

# API Helper with Retry Logic
def get_gpt_prediction(prompt, retries=3):
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_completion_tokens=150  # Fixed parameter name
            )
            ans = response.choices[0].message.content.strip().lower()
            return 1 if any(word in ans for word in ['yes', 'true', 'correct']) else 0
        except Exception as e:
            if i < retries - 1:
                time.sleep(2 ** i)
                continue
            print(f"Final error after retries: {e}")
            return -1

# Processing Loop
for ds in DATASETS:
    if not os.path.exists(ds):
        print(f"Skipping {ds}, file not found.")
        continue

    df = pd.read_csv(ds)
    print(f"--- Starting GPT-5.4 mini evaluation for {ds} ---")

    for strategy in STRATEGIES:
        start_time = time.time()
        print(f"Processing strategy {strategy}...", end=" ", flush=True)

        # apply() is convenient for smaller local runs
        df[f'gpt_{strategy}_pred'] = df['natural_language_triplet'].apply(
            lambda x: get_gpt_prediction(get_prompt(x, strategy))
        )

        elapsed = time.time() - start_time
        print(f"Done in {elapsed:.2f}s")

    output_name = f"results/results_gpt_{ds}"
    df.to_csv(output_name, index=False)
    print(f"Saved results to: {output_name}\n")