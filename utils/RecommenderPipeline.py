import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import requests
import re
import threading
import pickle

import concurrent.futures

import numpy as np
import pandas as pd
from functools import reduce
from collections import Counter

from sklearn.preprocessing import normalize, StandardScaler, Normalizer, RobustScaler, MinMaxScaler, MaxAbsScaler

import networkx as nx



import warnings
warnings.filterwarnings("ignore")

from url_utils import *
from wiki_scrapper import WikiScrapper
from WikiMultiQuery import wiki_multi_query
from graph_helpers import create_dispersion_df, sort_dict_values, format_categories, compare_categories, rank_order, similarity_rank

from GraphAPI import GraphCreator



# with open("../models/rf_classifier_v1.pkl", "rb") as model:
#     rf_classifier = pickle.load(model)

###############
# RECOMMENDER #
###############

class Recommender:

    def __init__(self, graph_creator, threads=20, chunk_size=1,):
        self.gc = graph_creator

        self.threads = threads
        self.chunk_size = chunk_size
        


    def fit(self, scaler=MinMaxScaler):
        self._expand_network()
        self._graph_cleanup()
        self._get_features()
        self._calculate_similarity()
        
        self.scaled = self._scale_features(scaler)

    def predict(self, model, size=100):
        
        self.classes = model.classes_

        X = self.scaled.drop(["node", "similarity_rank"], axis=1)[:size]
        preds = [list(x) for x in model.predict_proba(X)]

        result = self.scaled[["node", "similarity_rank"]][:size]
        result["label_proba"] = preds

        result = result[result.node != self.gc.entry]
        result_dict = result.to_dict(orient="index")
        self.predictions = [val for key, val in result_dict.items()]

    def format_results(self):
        for node in self.predictions:
            pred = node['label_proba'][0] > node['label_proba'][1]
            node['position'] = self.classes[0] if pred else self.classes[1]
        
        return {
            "entry": self.gc.entry,
            "predictions": self.predictions
        }

        

    def _expand_network(self):
        self.gc.expand_network_threaded(threads=self.threads, chunk_size=self.chunk_size)

    def _graph_cleanup(self):
        self.gc.redraw_redirects()
        self.gc.update_edge_weights()

    def _get_features(self):
        self.gc.get_features_df(rank=False)

    def _calculate_similarity(self):
        self.gc.rank_similarity()

    def _scale_features(self, scaler=MinMaxScaler):
        return self.gc.scale_features_df(scaler=scaler, 
            copy=True).sort_values("similarity_rank", 
            ascending=False).reset_index().drop("index", axis=1)

    

def predictor(entry, max_initial_nodes=500):

    gc = GraphCreator(entry)

    if len(gc.graph.nodes) > max_initial_nodes:
        return "Too Large"
    
    gc.expand_network_threaded(threads=20, chunk_size=1)
    gc.redraw_redirects()
    gc.update_edge_weights()
    features_df = gc.get_features_df(rank=False)
    gc.rank_similarity()
    scaled_feature_df = gc.scale_features_df(scaler=MinMaxScaler, copy=True) # Makes a copy of the df
    sorted_scaled = scaled_feature_df.sort_values("similarity_rank", ascending=False).reset_index().drop("index", axis=1)

    # drop the entry node row from recommendations
    # limit to first 100 recommendations
    sorted_scaled = sorted_scaled[sorted_scaled.node != gc.entry][0:100].reset_index().drop("index", axis=1)

    # format df for predictions
    X = sorted_scaled.drop(["node", "similarity_rank"], axis=1)

    y_preds = rf_classifier.predict_proba(X)

    classes = rf_classifier.classes_

    sorted_scaled['label'] = list(y_preds)

    results = sorted_scaled[["node", "label", "similarity_rank"]].to_dict(orient="index")


    return results




if __name__ == "__main__":
    prediction_pipeline("Prevention science")