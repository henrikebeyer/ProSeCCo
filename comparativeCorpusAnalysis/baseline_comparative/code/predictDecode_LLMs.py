import pandas as pd
import random
import os
import sys
from tqdm import tqdm
import ollama

MODEL = sys.argv[1]
DATASET_FILE = "/home/henrike/Self-Contradiction_project/data/decode_prosecco_size_df.tsv"
OUT_DIR = f"/home/henrike/Self-Contradiction_project/comparative_study/{MODEL}_preds"
df = pd.read_csv(DATASET_FILE, sep="\t")

os.makedirs(OUT_DIR, exist_ok=True)

# labels to strings
df["label"] = ["self-contradiction" if x == 1 else "no-self-contradiction" for x in df["label"]]

def query_ollama(prompt, model=MODEL):
    """Send prompt to Ollama and return model output text."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()

# === Build prompt with primes ===
def build_prompt(primes, premise, hypothesis):

    prompt = "You are a classification system trained to classify pairs of statements into 'self-contradiction' and 'no-self-contradiction. You can always assume that the statements are uttered by the same speaker. Only return the label.'\n\n"
    for i, (p, h, label) in enumerate(primes, 1):
        prompt += f"To help you, please consider the following examples: {i}:\nSentence1: \"{p}\"\nSentence2: \"{h}\"\nwould be classified as {label}\n\n"
    prompt += f"Now classify this sentence pair:\nSentence1: \"{premise}\"\nSentence2: \"{hypothesis}\""
    return prompt

# === Prime selection (done once per run) ===
def select_primes(k, seed):
    random.seed(seed)
    contra = df[df["label"] == "self-contradiction"]
    no_contra = df[df["label"] == "no-self-contradiction"]
    
    if k >= 2 and not contra.empty and not no_contra.empty:
        contra_ex = contra.sample(1, random_state=seed)
        no_contra_ex = no_contra.sample(1, random_state=seed)
        remaining = df.drop(pd.concat([contra_ex, no_contra_ex]).index)
        rest = remaining.sample(k - 2, random_state=seed) if k > 2 else pd.DataFrame()
        primes_df = pd.concat([contra_ex, no_contra_ex, rest])
    else:
        primes_df = df.sample(k, random_state=seed)

    cond = df.index.isin(primes_df.index)
    sample_df = df.drop(df[cond].index, inplace = False)

    return list(zip(primes_df["seg1"], primes_df["seg2"], primes_df["label"])), sample_df

# === Run a single experiment ===
def run_experiment(k, seed, run_id):
    primes, sample_df = select_primes(k, seed)
    preds = []
    for idx, row in tqdm(sample_df.iterrows(), total=len(df),
                         desc=f"k={k}, seed={seed}, run={run_id}"):
        prompt = build_prompt(primes, row["seg1"], row["seg2"])
        output = query_ollama(prompt)
        # Normalize output
        preds.append(output)
        # print(output)
    sample_df[f"{MODEL}_preds"] = preds
    return sample_df

# === Loop over all runs ===

for k in [0]:
    for run_id in range(1, 4):  # 3 repetitions
            pred_df = run_experiment(k, 42, run_id)
            pred_df.to_csv(f"{OUT_DIR}/{MODEL}_preds_decode_{k}-Shot_{run_id}.tsv", sep="\t", index=False)

for k in [2, 4, 8, 16, 32]:
    for seed in [42, 52, 121, 78]:
        for run_id in range(1, 4):  # 3 repetitions
            pred_df = run_experiment(k, seed, run_id)
            pred_df.to_csv(f"{OUT_DIR}/{MODEL}_preds_decode_{k}-Shot_{seed}_{run_id}.tsv", sep="\t", index=False)


# pred_df = run_experiment(4, 42, 1)

# print(pred_df)
# # Save results
# with open("llama_nli_results.json", "w") as f:
#     json.dump(results, f)