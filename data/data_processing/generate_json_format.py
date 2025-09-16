import pandas as pd
import json

TSV_PATH = "/home/oenni/Dokumente/Self-ContradictionProject/SelfContra-project/data/ProSeCCo_final_corpus/ProSeCCo_final.tsv"

JSON_PATH = "/home/oenni/Dokumente/Self-ContradictionProject/SelfContra-project/data/ProSeCCo_final_corpus/ProSeCCo_final.json"

def tsv_to_json(tsv_path, json_path):
    df = pd.read_csv(tsv_path, sep="\t")
    records = df.to_dict(orient='records')
    
    with open(json_path, 'w') as json_file:
        for record in records:
            json.dump(record, json_file)
            json_file.write('\n')

tsv_to_json(TSV_PATH, JSON_PATH)

def tsv_to_csv(tsv_path, csv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    df.to_csv(csv_path, index=False)

tsv_to_csv(TSV_PATH, TSV_PATH.replace('.tsv', '.csv'))