import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_validation_scores(df, entry):
    """
    given a validation score dataframe, plots the score and difference values in a barplot
    """


    # scores = df.score
    # diffs = df.difference
    top_1 = df['% targets in top 1%']
    top_5 = df['% targets in top 5%']
    top_10 = df['% targets in top 10%']
    top_20 = df['% targets in top 20%']

    ind = np.arange(len(top_1))
    width = 0.2

    fig, ax = plt.subplots(figsize=(16,6))

    ax.bar(ind - 0.3, top_1, width, color='#2166AC')
    ax.bar(ind - 0.1, top_5, width, color='#4393C3')
    ax.bar(ind + 0.1, top_10, width, color="#92C5DE")
    ax.bar(ind + 0.3, top_20, width, color="#D1E6F0")

    ax.set_facecolor("#F2F2F2")
    ax.set_xticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.index)

    plt.xticks(fontsize=16, rotation=70)
    plt.ylabel("% of 'See Also' links", fontsize=16)
    plt.legend([
        "top 1% of recs.",
        "top 5% of recs.",
        "top 10% of recs.",
        "top 20% of recs.",
    ], fontsize=14),

    plt.title(f"% 'See Also' links captured in top recs.\nby similarity score - {entry}", fontsize=26)

    plt.show()
