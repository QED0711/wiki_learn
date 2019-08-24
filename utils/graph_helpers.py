import pdb
import pandas as pd
import networkx as nx

def sort_dict_values(dict, columns, sort_column, ascending=False):
    to_list = [(key, value) for key, value in dict.items()]
    return pd.DataFrame(to_list, columns=columns).sort_values(sort_column, ascending=ascending).reset_index().drop("index", axis=1)

def create_dispersion_df(G, central_node, node_list):
    print("MAKING DISPERSION")
    dispersion = [(central_node, node, nx.dispersion(G, central_node, node)) for node in node_list]
    return pd.DataFrame(dispersion, columns=["entry", "node", "dispersion"]).sort_values("dispersion", ascending=False).reset_index().drop("index", axis=1)

def format_categories(cat_list):
    cat_dict = {}
    for cat in cat_list:
        cat_dict[cat] = True    
    return cat_dict

def compare_categories(node1, node2, categories):
    match_count = 1
    if node1 == node2:
        return 1
        
    try:
        for cat in categories[node1].keys():
            if categories[node2].get(cat):
                match_count += 1
        return match_count
    except:
        return 1