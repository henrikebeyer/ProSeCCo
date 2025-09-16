"""
This script shall parse the corpus json-files of QT30, QT50, and US2016
- it shall extract the L and I nodes connected by a Conflict node that have the same speaker
- it shall create one file separated by "#" per corpus containing these nodes with all their information
"""

import csv
import json
import pathlib
#import requests

qt30_path = pathlib.Path("../jsons/QT30")
us2016_path = pathlib.Path("../jsons/US2016")
qt50_path = pathlib.Path("../jsons/QT50")

corpus_list = [qt30_path, qt50_path, us2016_path]

"""
This function identifies the conflict nodes in a json file
Input: the parsed json-file
Output: a list of ids of conflict nodes
"""
def find_conflict(data):
    conflict_nodes = []
    for node in data["nodes"]:
                if "text" in node.keys():
                    if node["text"] == "Default Conflict":
                        confl_id = node["nodeID"]
                        conflict_nodes.append(confl_id)
    return conflict_nodes

"""
This function identifies the from nodes of a given node
Input: a node ID
Output: a list of dictionaries containing the from nodes
"""
def find_from_nodes(node_id, data):
    from_nodes = []
    for edge in data["edges"]:
         if edge["toID"] == node_id:
              fromID = edge["fromID"]
              for node in data["nodes"]:
                   if node["nodeID"] == fromID:
                        type = node["type"]
                        text = node["text"]
                        node_content = {"nodeID":fromID, "type":type, "text":text}
                        from_nodes.append(node_content)
    return from_nodes

"""
This function identifies the to nodes of a given node
Input: a node ID
Output: a list of dictionaries containing the to nodes
"""
def find_to_nodes(node_id, data):
    to_nodes = []
    for edge in data["edges"]:
         if edge["fromID"] == node_id:
              toID = edge["toID"]
              for node in data["nodes"]:
                   if node["nodeID"] == toID:
                        type = node["type"]
                        text = node["text"]
                        node_content = {"nodeID":toID, "type":type, "text":text}
                        to_nodes.append(node_content)
    return to_nodes

"""
This function identifies the I-node recursively in a maximum distance of 4 connecting nodes
Input: a list of node-dictionaries
Output: the node-dictionary of the I-node
"""
def find_i_node(node_list, data, i):
     for node in node_list:
          if node["type"] == "I":
               return node
          elif i == 4:
               return None
          else:
               from_nodes = find_from_nodes(node["nodeID"],data)
               return find_i_node(from_nodes, data,  i+1)


"""
This function identifies the L-node recursively in a maximum distance of 4 connecting nodes
Input: a list of node-dictionaries
Output: the node-dictionary of the L-node
"""
def find_l_node(node_list, data, i):
     for node in node_list:
          if node["type"] == "L":
               return node
          elif i == 4:
               return None
          else:
               from_nodes = find_from_nodes(node["nodeID"],data)
               return find_l_node(from_nodes, data,  i+1)

"""
This function checks whether the speaker of a pair of L-nodes is identical
Input: a pair of L-nodes
Output: a boolean indicating whether the speakers are identical
"""
def check_speaker(l_node_pair):
    #print(l_node_pair[0]["text"], "\n", l_node_pair[1]["text"])
    speaker1 = l_node_pair[0]["text"].split(":")[0]
    speaker2 = l_node_pair[1]["text"].split(":")[0]
    #print("speakers: ", speaker1, speaker2)
    if speaker1 == speaker2:
        #print("identic")
        return True
    else:
        return False

"""
This function:
- parses the json-files
- identifies the conflict nodes
- extracts the L and I nodes with the same speaker
- writes the full nodes to an output-file
"""
def create_conflict_csv(path, corpus_name):
    conflict_pairs = []

    for item in path.iterdir():

        # read in the json-file
        if str(item).endswith(".json"):
            try:
                with open(item, "r") as f:
                    data = json.load(f)

                    # find conflict nodes in the file
                    conflicts = find_conflict(data)

                    # find the from and to nodes for each conflict
                    for conflict in conflicts:
                        fromNodes = find_from_nodes(conflict, data)
                        toNodes = find_to_nodes(conflict, data)
                        conflict_l_pair = []
                        conflict_i_pair = []

                        # find the L and I nodes in the from nodes
                        for from_node in fromNodes:
                            #print(from_node)
                            if from_node["type"] == "L":
                                conflict_l_pair.append(from_node)
                            elif from_node["type"] == "I":
                                #print("here")
                                ancs = find_from_nodes(from_node["nodeID"], data)
                                #print("ancs", ancs)
                                l_node = find_l_node(ancs, data, 1)
                                #print("l_node", l_node)
                                if l_node != None:
                                    conflict_l_pair.append(l_node)

                        for from_node in fromNodes:
                            #print(from_node)
                            if from_node["type"] == "I":
                                conflict_i_pair.append(from_node)
                            else:
                                ancs = find_from_nodes(from_node["nodeID"], data)
                                #print("ancs", ancs)
                                i_node = find_i_node(ancs, data, 1)
                                #print("l_node", l_node)
                                if i_node != None:
                                    conflict_i_pair.append(i_node)
                        
                        # find the L and I nodes in the to nodes                
                        for to_node in toNodes:
                            if to_node["type"] == "L":
                                conflict_l_pair.append(to_node)
                            elif to_node["type"] == "I":
                                conflict_i_pair.append(to_node)
                                post = find_from_nodes(to_node["nodeID"], data)
                                #print("post", post)
                                l_node = find_l_node(post, data, 1)
                                #print("l_node", l_node)
                                if l_node != None:
                                    conflict_l_pair.append(l_node)

                        for to_node in toNodes:
                            if to_node["type"] == "I":
                                conflict_i_pair.append(to_node)
                            else:
                                post = find_from_nodes(to_node["nodeID"], data)
                                #print("post", post)
                                i_node = find_i_node(post, data, 1)
                                #print("l_node", l_node)
                                if i_node != None:
                                    conflict_i_pair.append(i_node)
                        

                        # only keep the nodes with the same speaker
                        if check_speaker(conflict_l_pair) == True:
                            if conflict_l_pair[0]["text"] !=  conflict_l_pair[1]["text"]:         
                                conflict_pairs.append([conflict_l_pair[0], conflict_l_pair[1], conflict_i_pair[0], conflict_i_pair[1], str(item).split("\\")[-1]])
                                #print(conflict_l_pair[0]["text"], "\n",  conflict_l_pair[1]["text"], "\n\n")
                                #print(conflict_i_pair[0]["text"], "\n", conflict_i_pair[1]["text"])

            # catch the unreadable json files
            except Exception:
                print(f"file could not be parsed by JSON Decoder")

    # remove potential duplicates from the conflict pairs
    clean_results = []

    for pair in conflict_pairs:
        if pair not in clean_results:
            clean_results.append(pair)


    # write te clean results to an output file
    with open(f"../raw_CorpusData_output/{corpus_name}_SelfConflict_i_and_l_all", "w", encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter = "#")
        writer.writerow(["ID", "L-node1", "L-node2", "I-node1", "I-node2", "file", "label"])

        for i in range(1, len(clean_results)):
            if i < 10:
                writer.writerow([f"{corpus_name}_00{i-1}"] + clean_results[i] + [""])
            elif i < 100:
                writer.writerow([f"{corpus_name}_0{i-1}"] + clean_results[i] + [""])
            else:
                writer.writerow([f"{corpus_name}_{i-1}"] + clean_results[i] + [""])

create_conflict_csv(qt30_path, "QT30")
create_conflict_csv(qt50_path, "QT50")
create_conflict_csv(us2016_path, "US2016")
