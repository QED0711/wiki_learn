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

def rank_order(df, rank_column, ascending=False):
    """
    Given dataframe with a column of numerical values, order and rank those values.
    Allows for ties if a value is the same as the previous value. 
    """
    df = df.sort_values(rank_column, ascending=ascending).reset_index().drop('index', axis=1)
    rankings = [1]
    for i in range(1, df.shape[0]):
        # pdb.set_trace()
        if df[rank_column][i] == df[rank_column][i-1]: # if value is same as last val
            rankings.append(rankings[-1])
        else: # if value is different from last val
            rankings.append(rankings[-1] + 1)

    df[f"{rank_column}_ranked"] = rankings
    return df