import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import scipy.cluster.hierarchy as sch


def _get_distance_matrix(Sigma: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distances matrix for Sigma correlation matrix: sqrt(1/2 * (1 - rho))
    """
    result = np.sqrt(0.5 * (1 - np.minimum(Sigma, 1.0)))
    result.iloc[range(len(Sigma)), range(len(Sigma))] = 0.0
    return result


def sort_corr(corr_matrix: pd.DataFrame) -> list[str]:
    """
    Sort correlation matrix for better visualization
    """
    corr_matrix = corr_matrix.copy()
    distance_matrix = _get_distance_matrix(corr_matrix)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        linkage_matrix = sch.complete(distance_matrix)
    cluster_labels = sch.fcluster(linkage_matrix, 6, criterion='maxclust')

    sorted_cols_and_orders = sorted(
        list(zip(corr_matrix.columns, cluster_labels)), key=lambda x: x[1]
    )
    sorted_cols, _ = list(map(list, list(zip(*sorted_cols_and_orders))))
    return sorted_cols


def plot_correlation_matrix(Sigma: pd.DataFrame, sorted_labels: list[str] | None = None) -> list[str]:
    """
    Sort correlation matrix and plot it
    """
    if sorted_labels is None:
        sorted_labels = sort_corr(Sigma)
    sns.heatmap(Sigma.loc[sorted_labels, sorted_labels], annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
    return sorted_labels
