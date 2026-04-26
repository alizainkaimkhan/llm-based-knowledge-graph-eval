import pandas as pd
from vllm import LLM, SamplingParams
import os

# Configuration
MODEL_PATH = "meta-llama/Meta-Llama-3-70B-Instruct"
DATASETS = ["data/FB15k-237_mapped_final.csv", "data/WN18RR_mapped_final.csv"]
STRATEGIES = ["S1", "S2", "S3", "S4"]

# Define 4 prompting strategies
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

# Execution Logic
def run_inference():
    # Initialize vLLM
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=2, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    for ds in DATASETS:
        if not os.path.exists(ds):
            print(f"Skipping {ds}, file not found.")
            continue

        df = pd.read_csv(ds)

        for strategy in STRATEGIES:
            print(f"Processing {ds} with strategy {strategy}...")
            prompts = [get_prompt(t, strategy) for t in df['natural_language_triplet']]

            outputs = llm.generate(prompts, sampling_params)

            preds = []
            for output in outputs:
                text = output.outputs[0].text.strip().lower()
                if any(word in text for word in ['yes', 'true', 'correct']):
                    preds.append(1)
                else:
                    preds.append(0)

            df[f'llama_{strategy}_pred'] = preds

        output_file = f"results/results_llama_{ds}"
        df.to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")

if __name__ == "__main__":
    run_inference()