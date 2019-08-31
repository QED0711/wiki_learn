import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import requests
import re
import threading

import concurrent.futures

import numpy as np
import pandas as pd
from functools import reduce
from collections import Counter


from sklearn.preprocessing import normalize, StandardScaler, Normalizer, RobustScaler, MinMaxScaler, MaxAbsScaler

import networkx as nx

import signal

import warnings
warnings.filterwarnings("ignore")

from url_utils import *
from wiki_scrapper import WikiScrapper
from WikiMultiQuery import wiki_multi_query
from graph_helpers import create_dispersion_df, sort_dict_values, format_categories, compare_categories, rank_order, similarity_rank


################
# GraphCreator #
################

class GraphCreator:

    def __init__(self, entry, include_see_also=True, max_recursize_requests=50):
        self.graph = nx.DiGraph()

        self.entry = get_title(entry) # from url_utils

        self.max_requests = max_recursize_requests

        ws = WikiScrapper(f"https://en.wikipedia.org/wiki/{self.entry}")
        ws.parse_intro_links()

        self.primary_nodes = {title : True for title in ws.get_primary_links(include_see_also=include_see_also)}

        # see also articles to be used as targets for evaluation
        self.see_also_articles = ws.see_also_link_titles

        self.visited = {self.entry}
        self.next_links = []
        
        self.categories = {}
        
        self.redirect_targets = []
        self.redirect_sources = {}
        
        self.query_articles([self.entry])

        # setup timeout function

        def handle_alarm(signum, frame):
            raise RuntimeError

        signal.signal(signal.SIGALRM, handle_alarm)

    def add_edges(self, articles):

        for article in articles:          
            self.categories[article['title']] = format_categories([category.split("Category:")[1] for category in article['categories'] if not bool(re.findall(r"(articles)|(uses)|(commons)|(category\:use)", category, re.I))])
            
            self.graph.add_edges_from(
                [(article['title'], link) for link in article['links']])
            self.graph.add_edges_from(
                [(linkhere, article['title']) for linkhere in article['linkshere']])

    def update_edge_weights(self):
        for edge in self.graph.out_edges:
            weight = compare_categories(edge[0], edge[1], self.categories)
            self.graph.add_edge(edge[0], edge[1], weight=weight)
            
        for edge in self.graph.in_edges:
            weight = compare_categories(edge[0], edge[1], self.categories)
            self.graph.add_edge(edge[0], edge[1], weight=weight)

    def get_edge_weights(self):
        edge_weights = []
        for edge in self.graph.edges:
            edge_weights.append((edge[0], edge[1], self.graph.get_edge_data(edge[0], edge[1])['weight']))
        
        return pd.DataFrame(edge_weights, columns=["source_node", "target_node", "edge_weight"]).sort_values("edge_weight", ascending=False).reset_index().drop("index", axis=1)
    
    def plot_graph(self):
        nx.draw(self.graph)
        plt.show()

    def get_shared_categories_with_source(self):
        cat_matches = {}
        for node in self.graph.nodes:
            cat_matches[node] = compare_categories(self.entry, node, self.categories, starting_count=0)
        return sort_dict_values(cat_matches, ['node', 'category_matches_with_source'], 'category_matches_with_source', ascending=False)
            
    
    def get_primary_nodes(self):
        primary_nodes = {}
        for node in self.graph.nodes:
            if node in primary_nodes:
                # allows for heavier weight to duplicates in intro and see also
                primary_nodes[node] += 1
            
            if node in self.primary_nodes:
                primary_nodes[node] = 1
            else: 
                primary_nodes[node] = 0
        return sort_dict_values(primary_nodes, ["node", "primary_link"], "primary_link", ascending=False)

    def get_degrees(self):
        return sort_dict_values(dict(self.graph.degree()), ["node", "degree"], "degree",)

    def get_shared_neighbors_with_entry_score(self):
        """
        A score comprised of the total number of shared neighbors with the target OVER the total number of neighbors
        of each node 
        """
        entry_neighbors = list(set(nx.all_neighbors(self.graph, self.entry)))
        shared_neighbors_score = {}
        for node in self.graph.nodes:
            target_neighbors = list(set(nx.all_neighbors(self.graph, node)))
            shared_neighbors = len(entry_neighbors) + len(target_neighbors) - len(set(entry_neighbors + target_neighbors))
            # score is neighbors shared over how many possible unique neighbors could have been shared. 
            shared_neighbors_score[node] = shared_neighbors / len(set(entry_neighbors + target_neighbors))

        return sort_dict_values(shared_neighbors_score, ["node", "shared_neighbors_with_entry_score"], "shared_neighbors_with_entry_score", ascending=False)

    def get_edges(self):
        edges = []
        for node in self.graph.nodes:
            node_in_edges = len(self.graph.in_edges(node))
            node_out_edges = len(self.graph.out_edges(node))
            edges.append({"node": node, "in_edges": node_in_edges, "out_edges": node_out_edges})
            
        return pd.DataFrame(edges)
    
    def get_centrality(self):
        return sort_dict_values(nx.eigenvector_centrality(self.graph, weight="weight"), ["node", "centrality"], "centrality")

    def get_dispersion(self, comparison_node=None, max_nodes=25_000): # depreciated
        if not comparison_node:
            comparison_node = self.entry
            
        if max_nodes is None or len(self.graph.nodes) <= max_nodes:
            return sort_dict_values(nx.dispersion(self.graph, u=comparison_node), ['node', 'dispersion'], 'dispersion')
        else:
            # if the network is too large, perform calculation on ego graph of entry node
            ego = self.create_ego()
            return sort_dict_values(nx.dispersion(ego, u=comparison_node), ['node', 'dispersion'], 'dispersion')

    def get_pageranks(self):
        page_ranks = sorted([(key, value) for key, value in nx.algorithms.link_analysis.pagerank(
            self.graph, weight='weight').items()], key=lambda x: x[1], reverse=True)
        return pd.DataFrame(page_ranks, columns=["node", "page_rank"])

    def get_reciprocity(self):
        return sort_dict_values(nx.algorithms.reciprocity(self.graph, self.graph.nodes), ['node', 'reciprocity'], 'reciprocity')

    def get_adjusted_reciprocity(self):
        r = self.get_reciprocity()
        d = self.get_degrees()

        r_d = r.merge(d, on="node", how="inner")
        r_d['adjusted_reciprocity'] = r_d.reciprocity * r_d.degree

        adjusted_reci = r_d.sort_values("adjusted_reciprocity", ascending=False)
        adjusted_reci.adjusted_reciprocity = normalize([adjusted_reci.adjusted_reciprocity])[0]
        return adjusted_reci.reset_index().drop(["degree", "reciprocity", "index"], axis=1)
    
    def get_shortest_path_from_entry(self):
        paths = []
        for node in self.graph.nodes:
            try:
                path_length = nx.shortest_path_length(self.graph, source=self.entry, target=node) 
                paths.append({"node": node, "shortest_path_length_from_entry": path_length})
            except:
                paths.append({"node": node, "shortest_path_length_from_entry": np.nan})
                

        from_entry = pd.DataFrame(paths).sort_values("shortest_path_length_from_entry", ascending=False)
        return from_entry.fillna(np.max(from_entry.shortest_path_length_from_entry) + 1)  

    def get_shortest_path_to_entry(self):
        paths = []
        for node in self.graph.nodes:
            try:
                path_length = nx.shortest_path_length(self.graph, source=node, target=self.entry) 
                paths.append({"node": node, "shortest_path_length_to_entry": path_length})
            except:
                paths.append({"node": node, "shortest_path_length_to_entry": np.nan})
                

        to_entry = pd.DataFrame(paths)
        return to_entry.fillna(np.max(to_entry.shortest_path_length_to_entry) + 1)

    def get_jaccard_similarity(self):
        entry_in_edges = set([x[0] for x in self.graph.in_edges(nbunch=self.entry)])
        jaccard_scores = {}
        for node in self.graph.nodes:
            target_in_edges = set([x[0] for x in self.graph.in_edges(nbunch=node)])
            in_edge_intersect = len(entry_in_edges.intersection(target_in_edges))
            in_edge_union = len(entry_in_edges.union(target_in_edges))
            
            jaccard_scores[node] = in_edge_intersect / in_edge_union


        return sort_dict_values(jaccard_scores, ["node", "jaccard_similarity"], "jaccard_similarity", ascending=False)


    def get_dominator_counts(self, source=None):
        if not source:
            source = self.entry
            
        dom_dict = nx.algorithms.dominance.immediate_dominators(self.graph, start=source)
        
        dom_counts = {}

        for key, value in dom_dict.items():
            if value in dom_counts:
                dom_counts[value] += 1
            else:
                dom_counts[value] = 1
        for node in self.graph.nodes:
            if not node in dom_counts:
                dom_counts[node] = 0
        
        return sort_dict_values(dom_counts, ['node', 'immediate_dominator_count'], 'immediate_dominator_count')

    def get_hits(self):
        hits = nx.algorithms.link_analysis.hits_alg.hits(self.graph, max_iter=1000)
        return (sort_dict_values(hits[1], ['node', 'hits_authority'], 'hits_authority')
                .merge(sort_dict_values(hits[0], ['node', 'hits_hub'], 'hits_hub'), on="node"))
    
    def get_features_df(self, rank=False):
        dfs = []
        if rank:
            dfs.append(rank_order(self.get_degrees(), 'degree', ascending=False))
            dfs.append(rank_order(self.get_shared_categories_with_source(), 'category_matches_with_source', ascending=False))
            dfs.append(rank_order(self.get_shared_neighbors_with_entry_score(), 'shared_neighbors_with_entry_score', ascending=False))
            dfs.append(self.get_edges())
            dfs.append(rank_order(self.get_centrality(), 'centrality', ascending=True))
            # dfs.append(rank_order(self.get_dispersion(), "dispersion", ascending=True))
            dfs.append(rank_order(self.get_pageranks(), "page_rank", ascending=False))
            dfs.append(rank_order(self.get_adjusted_reciprocity(), "adjusted_reciprocity", ascending=False))
            dfs.append(rank_order(self.get_shortest_path_from_entry(), "shortest_path_length_from_entry", ascending=True))
            dfs.append(rank_order(self.get_shortest_path_to_entry(), "shortest_path_length_to_entry", ascending=True))
            dfs.append(rank_order(self.get_jaccard_similarity(), "jaccard_similarity", ascending=False))
            dfs.append(rank_order(self.get_primary_nodes(), "primary_node", ascending=False))
        
        else:
            dfs.append(self.get_degrees())
            dfs.append(self.get_shared_categories_with_source())
            dfs.append(self.get_edges())
            dfs.append(self.get_shared_neighbors_with_entry_score())
            dfs.append(self.get_centrality())
            # dfs.append(self.get_dispersion())
            dfs.append(self.get_pageranks())
            dfs.append(self.get_adjusted_reciprocity())
            dfs.append(self.get_shortest_path_from_entry())
            dfs.append(self.get_shortest_path_to_entry())
            dfs.append(self.get_jaccard_similarity())
            dfs.append(self.get_primary_nodes())
        
        self.features_df = reduce(lambda left, right: pd.merge(left, right, on="node", how="outer"), dfs)
        return self.features_df

    def rank_similarity(self):
        degree_mean = np.mean(self.features_df.degree.unique())

        self.features_df['similarity_rank'] = self.features_df.apply(
                                                    similarity_rank, 
                                                    degree_mean=degree_mean,
                                                    axis=1)
    
    def scale_features_df(self, scaler=StandardScaler, copy=True):
        nodes = self.features_df.node

        node_removed = self.features_df.drop("node", axis=1)
        columns = node_removed.columns 

        scaled_features = scaler().fit_transform(node_removed)

        scaled_features = pd.DataFrame(scaled_features, columns=columns)

        scaled_features.insert(0, "node", nodes)
        
        if not copy: # if copy=False, then we overwrite existing features_df with the scaled version
            self.features_df = scaled_features
            return self.features_df
        else:
            return scaled_features


    def create_ego(self, node=None):
        if not node:
            node = self.entry

        ego = nx.ego_graph(self.graph, node)
        ego.name = node
        return ego

    def _next_link_chunks(self, chunk_size=2):
        chunked = []
        current_chunk = []
        for i, node in enumerate(self.graph.nodes):
            if not node in self.visited:
                current_chunk.append(node)
                self.visited.update([node])
            if len(current_chunk) == chunk_size or (i == len(self.graph.nodes) - 1 and len(current_chunk) > 0):
                chunked.append(current_chunk)
                current_chunk = []
        return chunked


    def expand_network(self, group_size=10, timeout=10, log_progress=False):

        num_links = len(self.next_links)

        link_group = []

        for i in range(num_links):
            link = self.next_links.pop(0)
            if not link in self.visited:

                link_group.append(link)

                if len(link_group) == group_size or (i == num_links - 1 and len(link_group) > 0):
                    print("{:.2%}".format(i/num_links)) if log_progress else None
                    try:
                        signal.alarm(timeout)
                        self.visited.update(link_group)
                        self.query_articles(link_group)
                        signal.alarm(0)
                        link_group = []
                    except:
                        link_group = []
                        continue
        signal.alarm(0)

    def expand_network_threaded(self, threads=10, chunk_size=5):
        """
        A wrapper method that handles multithreaded network expansion.
        The primary purpose is to call the `_run_threaded_expansion` method.

        This method can be substantially more performant than the non threaded `expand_network` method. 
        The trade off is that you need to be careful about API rate limits, as this can easily exceed them. 
        It is best to set the `threads` parameter to a reasonable number (2-20) so as to avoide rate limit issues. 

        Input:
        ------

        threads (default: 10)
        the number of concurrent threads to run.

        chunk_size (default: 5)
        The number of articles to query per api call. Setting this higher will reduce the total number of api calls, while making each call take slightly longer.
        """
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            executor.map(self._run_threaded_expansion, self._next_link_chunks(chunk_size))

    def _run_threaded_expansion(self, nodes):
        """
        A multithreaded network expansion helper function. Not to be called manually. 
        """
        try:
            signal.alarm(10)
            self.query_articles(nodes)
            signal.alarm(0)
        except:
            signal.alarm(0)
        return


    def update_redirects(self, articles):
        for article in articles:
            if article.get("redirects"):
                self.redirect_targets.append(article["title"])
                for redirect in article["redirects"]:
                    self.redirect_sources[redirect] = len(self.redirect_targets) - 1
    
    def redraw_redirects(self):
        edges = list(self.graph.edges) # need this copy so 'edges' doesn't change size on iteration
        for edge in edges:
            if edge[0] in self.redirect_sources:
                self.graph.add_edge(self.redirect_targets[self.redirect_sources[edge[0]]], edge[1])
                
            if edge[1] in self.redirect_sources:
                self.graph.add_edge(edge[0], self.redirect_targets[self.redirect_sources[edge[1]]])
        
        self.remove_redirect_nodes()
    
    def remove_redirect_nodes(self):
        nodes = list(self.graph.nodes) # need this copy so 'nodes' doesn't change size on iteration
        for node in nodes:
            if node in self.redirect_sources:
                self.graph.remove_node(node)
    
    def update_next_links(self, articles):
        for article in articles:
            out_in = article['links'] + article['linkshere']
            self.next_links += out_in

    def query_articles(self, titles, generate_graph=True):
        articles = wiki_multi_query(titles, max_requests=self.max_requests)
        
        self.update_redirects(articles)
        
        self.update_next_links(articles)
        self.add_edges(articles)


if __name__ == "__main__":
    gc = GraphCreator("Prevention science")
    print(len(gc.graph.nodes))
    # gc.expand_network(group_size=2, timeout=5, log_progress=True)
    gc.expand_network_threaded()
    print(len(gc.graph.nodes))