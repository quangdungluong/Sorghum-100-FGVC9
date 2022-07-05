"""Data Preparation
"""
import pandas as pd

from src.config import CFG


def create_labels_map(input_csv=CFG.input_csv):
    """Create labels map from input.csv

    Args:
        input_csv (.csv, optional): Train csv. Defaults to CFG.input_csv.

    Returns:
        labels_map: dictionary
    """
    dataframe = pd.read_csv(input_csv, index_col='image')
    labels_map = {}
    for i, label in enumerate(dataframe['cultivar'].unique()):
        labels_map[i] = label
    return labels_map
