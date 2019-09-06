import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_validation_scores(df):
    sns.barplot(data=df, x=df['Metric Score'], y="score")
