"""
This script runs the GPT priming experiment on the primes and samples created in GPT_priming_dataPrep.py
The output are .tsv files with the columns ID, sample, gold-standard, and prediction
ATTENTION: You will need your own GPT-API key to run this script
"""

import pandas as pd
from openai import OpenAI
client = OpenAI()

# please re-run this script as many times as you wish for validation purposes and manipulate this number if applicable
run = 1
sent_types = ["locution", "proposition"] # "proposition" "locution"
seeds = [52, 42, 121, 78]
inds = [2, 8, 16, 32]


messages_list = [[{"role": "system", "content": "You are a classification system classifying pairs of statements to be 'self-contradiction' or 'non-contradiction', please assume that the authors of both statements are identical"}],
            [{"role": "system", "content": "You are a classification system classifying pairs of statements to be 'self-contradiction' or 'non-contradiction', please assume that the authors of both statements are identical"}],
            [{"role": "system", "content": "You are a classification system classifying pairs of statements to be 'self-contradiction' or 'non-contradiction', please assume that the authors of both statements are identical"}],
            [{"role": "system", "content": "You are a classification system classifying pairs of statements to be 'self-contradiction' or 'non-contradiction', please assume that the authors of both statements are identical"}]
    ]

token_sum = 0
for seed in seeds:
    for sent_type in sent_types:
        preds = [[],[],[],[]]
        for i in range(len(inds)):
            samp_num = inds[i]
            print(samp_num)
            
            prime_df = pd.read_csv(f"~/Dokumente/SelfContra/baselines/GPT_priming/primes/{sent_type}_primes_{samp_num}_{seed}.tsv", sep="\t")
            sample_df = pd.read_csv(f"~/Dokumente/SelfContra/baselines/GPT_priming/samples/{sent_type}_samples_{samp_num}_{seed}.tsv", sep="\t")

            primes = list(prime_df["prime_sample"])
            prime_labels = list(prime_df["label"])

            samples = list(sample_df["test_sample"])
            sample_labels = list(sample_df["label"])

            prime_prompt = "To help you, please consider the following examples: \n"
            for prime, label in zip(primes, prime_labels):
                prime_prompt += f"{prime} would be classified as {label} \n"

            messages_list[i].append({"role": "system", "content": prime_prompt})

            for sample in samples:
                sample_prompt = f"Please classify the following sentence pair: \n {sample}"
                messages_list[i].append({"role": "system", "content": sample_prompt})

            for j in range(2,len(messages_list[i])):
                #print(i, j)
                mess = messages_list[i][:2] + [messages_list[i][j]]
                #print(mess[-1]["content"])
                #for mes in mess:
                    #print(mes["content"].split())
                    #token_sum += len(mes["content"].split())
                
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages= mess
                )

                preds[i].append(completion.choices[0].message.content)

            out_df = pd.DataFrame()
            out_df["ID"] = sample_df["ID"]
            out_df["sample"] = samples
            out_df["gold_standard"] = sample_labels
            out_df["prediction"] = preds[i]

            out_df.to_csv(f"predictions/gpt3_{sent_type}_predictions_{samp_num}_{seed}_{run}.tsv", sep="\t", index=False)

