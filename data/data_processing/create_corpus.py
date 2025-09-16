"""
This script parses the raw corpus files created by json-to-rawCorpus.py and transforms them to the final un-annotated corpus version
The output is a file containing all pairs of L and I node texts in all sub-corpora
"""

import pandas as pd
import ast

US2016_i_l_all = "../raw_CorpusData_output/US2016_SelfConflict_i_and_l_all"

QT30_i_l_all = "../raw_CorpusData_output/QT30_SelfConflict_i_and_l_all"

QT50_i_l_all = "../raw_CorpusData_output/QT50_SelfConflict_i_and_l_all"

annotated = "/home/oenni/Dokumente/Self-ContradictionProject/Additional_Material_Coling/ProSeCCo/fullCorpus/ProSeCCo_final.tsv"

merged = "../ProSeCCo_final_corpus/ProSeCCo_final.tsv"

labels = "/home/oenni/Dokumente/Self-ContradictionProject/SelfContra-project/data/ProSeCCo_final_corpus/ProSeCCo_final_mine.tsv"

"""
This function extracts the node text for all I and L nodes in a raw dataset file
Input: a dataset, a separator, and the corpus name
Output: a pandas data-frame of the dataset with the columns: ID, locution1, locution2, proposition1, proposition2, label, prop_needed
"""
def extract_text(dataset, sep, corpus_name):

    speaker_dict = {}
    in_df = pd.read_csv(dataset, sep=sep)
    speakers = [ast.literal_eval(node)["text"].split(":")[0].strip() for node in in_df["L-node1"]]
    nodesets = [file.split("/")[-1].split(".")[0].split("set")[-1] for file in in_df["file"]]

    l1 = [ast.literal_eval(node)["text"].split(":")[1].strip() for node in in_df["L-node1"]]
    l2 = [ast.literal_eval(node)["text"].split(":")[1].strip() for node in in_df["L-node2"]]
    i1 = [ast.literal_eval(node)["text"] for node in in_df["I-node1"]]
    i2 = [ast.literal_eval(node)["text"] for node in in_df["I-node2"]]

    i = 1
    for speaker in speakers:
        if speaker not in speaker_dict:
            if i < 10:
                speaker_dict[speaker] = f"speaker_00{i}"
            elif i > 9 and i < 100:
                speaker_dict[speaker] = f"speaker_0{i}"
            else:
                speaker_dict[speaker] = f"speaker_{i}"
            i += 1

    out_df = pd.DataFrame()
    out_df["ID"] = in_df["ID"]
    out_df["speaker_id"] = [speaker_dict[speaker] for speaker in speakers]
    out_df["locution_1"] = l1
    out_df["locution_2"] = l2
    out_df["proposition_1"] = i1
    out_df["proposition_2"] = i2
    out_df["label"] = ["" for loc in l1]
    out_df["source"] = [corpus_name for loc in l1]
    out_df["nodeset_id"] = nodesets
    out_df["prop_needed"] = ["" for loc in l1]

    return out_df

# create sub-corpora pandas dfs
QT30_df = extract_text(QT30_i_l_all, sep="#", corpus_name="QT30")
QT50_df = extract_text(QT50_i_l_all, sep="#", corpus_name="QT50")
US2016_df = extract_text(US2016_i_l_all, "#", "US2016")

# concatenate sub-corpora dfs and write them to a .tsv-file
large_corp = pd.concat([QT30_df, QT50_df, US2016_df])

# print(large_corp.head())
# large_corp.to_csv("../CAPTURE_noAnnotations/CAPTURE_ordered_noAnnotations.tsv", sep="\t", index=False)

def merge_with_annotations(corpus_df, annotations, labels, output_path):
    annotations_df = pd.read_csv(annotations, sep="\t").rename(columns={"locution1": "locution_1", "locution2": "locution_2", "proposition1": "proposition_1", "proposition2": "proposition_2", "prep_needed": "prop_needed", "label": "label", "source": "source", "topic": "topic"})
    labels = pd.read_csv(labels, sep="\t").rename(columns={"locution1": "locution_1", "locution2": "locution_2", "proposition1": "proposition_1", "proposition2": "proposition_2", "prep_needed": "prop_needed", "label": "label", "source": "source", "topic": "topic"})
    print(annotations_df["label"].value_counts())

    merged_df = pd.merge(corpus_df, annotations_df, on=["proposition_1", "proposition_2", "locution_1", "locution_2"], how="left").drop(columns=["label_x", "ID_y", "prop_needed_x"]).rename(columns={"label_y": "label", "ID_x": "ID", "prop_needed_y": "prop_needed"})

    print(merged_df["label"].value_counts())
    merged_df["label"] = labels["label"]
    print(merged_df["label"].value_counts())
    #print(annotations_df.head())

    print(merged_df.head())

    merged_df.to_csv(output_path, sep="\t", index=False)

merge_with_annotations(large_corp, annotated, labels, "../ProSeCCo_final_corpus/ProSeCCo_final.tsv")

# merged_df = pd.read_csv(merged, sep="\t")
# print(merged_df["label"].value_counts())