import pdb
import pandas as pd
import networkx as nx

def create_dispersion_df(G, central_node, node_list):
    dispersion = [(central_node, node, nx.dispersion(G, central_node, node)) for node in node_list]
    return pd.DataFrame(dispersion, columns=["entry", "node", "dispersion"])


