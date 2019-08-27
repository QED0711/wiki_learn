
import pdb
import numpy as np
import pandas as pd

from math import ceil

def score_at_percentage(alpha, df, targets):
    segment = ceil(alpha * df.shape[0])
    segmented_df = df[0:segment]
    targets_seen = 0
    for i, row in segmented_df.iterrows():
        if row.node in targets:
            targets_seen += 1
    
    return targets_seen


def evaluate_recommendations(df, on, targets):
    """
    Evaluates recommendation results based on how highly they rated the target values

    Input
    -----
    df (pandas dataframe): a dataframe containing the features and node names

    on (string): the column name that will be sorted on (how we rank our nodes/targets)

    targets (array-like): a list of known recommendations (from the "see also" section)


    return
    ------
    """
    sorted_df = df.sort_values(on, ascending=False).reset_index().drop("index", axis=1) 
    total_nodes = sorted_df.shape[0]

    max_target_index = 0

    for target in targets:
        try:
            target_val = sorted_df[sorted_df.node == target].index[0]
            if max_target_index < target_val:
               max_target_index = target_val 
        except:
            continue

    # Represents the percentage of rows we must go down before we have captured all targets
    # a higher number indicates that the targets are closer to the top of our recommendations
    score = 1 - (max_target_index / (total_nodes - len(targets)))
    max_score_possible = 1 - (len(targets) / total_nodes)
    report = pd.DataFrame([
        {"description": "score", "value": score},
        {"description": "% targets in top 1%", "value": (score_at_percentage(0.01, sorted_df, targets) / len(targets))},
        {"description": "% targets in top 5%", "value": (score_at_percentage(0.05, sorted_df, targets) / len(targets))},
        {"description": "% targets in top 10%", "value": (score_at_percentage(0.10, sorted_df, targets) / len(targets))},
        {"description": "% targets in top 20%", "value": (score_at_percentage(0.20, sorted_df, targets) / len(targets))},
        {"description": "max score posible", "value": max_score_possible},
        {"description": "difference", "value": max_score_possible - score},
        {"description": "total targets", "value": len(targets)},
    ]).set_index("description")

    return report
