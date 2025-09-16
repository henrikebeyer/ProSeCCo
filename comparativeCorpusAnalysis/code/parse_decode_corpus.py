import pandas as pd
from itertools import combinations

decode_train = "/home/oenni/Dokumente/Self-ContradictionProject/DECODE_Material_Download/decode_v0.1/train.jsonl"
decode_dev = "/home/oenni/Dokumente/Self-ContradictionProject/DECODE_Material_Download/decode_v0.1/dev.jsonl"
decode_test = "/home/oenni/Dokumente/Self-ContradictionProject/DECODE_Material_Download/decode_v0.1/test.jsonl"

train_df = pd.read_json(decode_train, lines=True)
dev_df = pd.read_json(decode_dev, lines=True)
test_df = pd.read_json(decode_test, lines=True)


def extract_contradictions(df):
    turns = df["turns"]
    contradiction_indices = df["aggregated_contradiction_indices"]

    contradictions = []

    for turn_list, index_list in zip(turns, contradiction_indices):
        if index_list != []:
            seg2_id = index_list[-1]
            #print(seg2_id)
            seg2 = turn_list[seg2_id]
            for ind in index_list[:-1]:
                seg1 = turn_list[ind]

                contradictions.append({"seg1":seg1["text"],
                                    "seg2":seg2["text"],
                                    "label":0})

    contra_df = pd.DataFrame(contradictions)
    return contra_df

def extract_non_contradictions(df):
    turns = df.loc[train_df["is_contradiction"]==False]["turns"]

    non_contradictions = []
    for turn_list in turns:
        agent_turns = {0:[],
                    1:[]}
        for turn in turn_list:
            agent_turns[turn["agent_id"]].append(turn["text"])

        for agent_turn_l in agent_turns.values():
            if len(agent_turn_l) > 1:
                subsets = [list(subs) for subs in list(combinations(agent_turn_l, 2))]
                for subs in subsets:
                    non_contradictions.append({"seg1":subs[0],
                                            "seg2":subs[1],
                                            "label":1})
            
    non_contra_df = pd.DataFrame(non_contradictions)
    return non_contra_df

decode_dfs = [train_df, dev_df, test_df]
contra_dfs = []
non_contra_dfs = []

print(train_df.columns)
print(train_df["is_contradiction"].unique())


for decode_df in decode_dfs:
    contra_dfs.append(extract_contradictions(decode_df))
    non_contra_dfs.append(extract_non_contradictions(decode_df))


full_contra_df = pd.concat(contra_dfs, copy=False)
full_non_contra_df = pd.concat(non_contra_dfs, copy=False)

contra_685 = full_contra_df.sample(n=685, random_state=42)
non_contra_640 = full_non_contra_df.sample(n=640, random_state=42)

prosecco_size = pd.concat([contra_685, non_contra_640], copy=False)

prosecco_size = prosecco_size.sample(frac=1, random_state=123)

prosecco_size.to_csv("/home/oenni/Dokumente/Self-ContradictionProject/SelfContra-project/comparativeCorpusAnalysis/decode_parsed/decode_prosecco_size_df.tsv", sep="\t", index=False)

# full_df.to_csv("/home/oenni/Dokumente/Self-ContradictionProject/decode_parsed/contradiction_df.tsv", sep="\t", index=False)

# contradictions_10 = full_df.sample(frac=0.1, random_state=42)

# contradictions_10.to_csv("/home/oenni/Dokumente/Self-ContradictionProject/decode_parsed/contradiction_df_random_10_perc.tsv", sep="\t", index=False)