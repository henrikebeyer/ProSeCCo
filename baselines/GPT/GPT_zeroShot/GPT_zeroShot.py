import pandas as pd
from openai import OpenAI
client = OpenAI()

ty = "locution" # "locution" "proposition"

preds = []

messages = [{"role": "system", "content": "You are a classification system classifying pairs of statements as 'self-contradiction' or 'non-contradiction', please assume that the authors of both statements are identical."}]

data = pd.read_csv(f"~/Dokumente/SelfContra/SelfContra_corpus/SelfContra_final.tsv", sep="\t")

statements1 = list(data[f"{ty}1"])
statements2 = list(data[f"{ty}2"])
labels = ["non-contradiction" if label == 0  else "self-contradiction" for label in list(data["label"])]
ids = list(data["ID"])

samples = []
 
for i in range(len(statements1)):
    statement1 = statements1[i]
    statement2 = statements2[i]
    statement_pair = f"'{statement1}', '{statement2}'"
    samples.append(statement_pair)
#samples = list(sample_df["test_sample"])
sample_labels = list(data["label"])


for sample in samples:
    sample_prompt = f"Please classify the following sentence pair: \n {sample}"
    messages.append({"role": "system", "content": sample_prompt})

input_tokens = 0
for message in messages:
    input_tokens += len(message["content"])

print(input_tokens)

for j in range(1,len(messages)):
    mess = [messages[0]] + [messages[j]]
    #print(mess[-1]["content"])
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages= mess
    )

    preds.append(completion.choices[0].message.content)

out_df = pd.DataFrame()
out_df["ID"] = data["ID"]
out_df["sample"] = samples
out_df["gold_standard"] = sample_labels
out_df["prediction"] = preds

out_df.to_csv(f"GPT_zeroShot_{ty}_predictions_3.tsv", sep="\t", index=False)

#for messages in messages_list:
#    for i in range(10):
#        print(messages[i]["content"])


