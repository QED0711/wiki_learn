import pdb
import pandas as pd
import networkx as nx

def sort_dict_values(dict, columns, sort_column, ascending=False):
    to_list = [(key, value) for key, value in dict.items()]
    return pd.DataFrame(to_list, columns=columns).sort_values(sort_column, ascending=ascending).reset_index().drop("index", axis=1)

def create_dispersion_df(G, central_node, node_list):
    dispersion = [(central_node, node, nx.dispersion(G, central_node, node)) for node in node_list]
    return pd.DataFrame(dispersion, columns=["entry", "node", "dispersion"])




