from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from tree_node import CalibrationTree, key_to_name


def extreme_outliers_mad(data: np.ndarray, threshold: float=4.25) -> np.ndarray:
    """
    Identify extreme outliers using the Median Absolute Deviation (MAD) method.
    :param data: Input data array
    :param threshold: Threshold for modified Z-score to consider as outlier
    :return: Indices of extreme outliers
    """
    data = np.array(data)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return np.array([], dtype=int)  # No variation = no outliers
    modified_z = 0.6745 * (data - median) / mad
    return np.where(np.abs(modified_z) > threshold)[0]  # returns indices of extreme outliers


def remove_outliers(per_node_train_dfs: Dict[str, pd.DataFrame], score_col: str):
    """
    Remove extreme outliers from per-node training dataframes.
    :param per_node_train_dfs: A dictionary mapping node keys to their training DataFrames
    :param score_col: The name of the score column to check for outliers
    :return: A new dictionary of DataFrames with outliers removed
    """
    dfs = []
    for node_key, df in per_node_train_dfs.items():
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=False)  # keep original indices

    # Detect outliers
    outliers_idxs = extreme_outliers_mad(all_data[score_col].to_numpy(), threshold=4.25)

    if outliers_idxs.size > 0:
        mask = np.ones(len(all_data), dtype=bool)
        mask[outliers_idxs] = False
        all_data = all_data.loc[mask].copy()

    # Get indices that remain
    valid_indices = set(all_data.index)

    # Rebuild per-node dataframes with original indices preserved
    new_per_node_train_dfs = {}
    for node_key, df in per_node_train_dfs.items():
        new_per_node_train_dfs[node_key] = df.loc[df.index.intersection(valid_indices)].copy()

    return new_per_node_train_dfs


def generate_gmm_key_from_calibration_tree(calibration_tree: CalibrationTree, leaf_key: str) -> str:
    """
    Generate GMM key from CalibrationTree and leaf_key.
    :param calibration_tree: The calibration tree
    :param leaf_key: The current leaf key string
    :return: A string representing the GMM key. This string encodes the combination of features
             in a binary format, where each feature is represented by '0', '1', or '2' based on its value.
    """
    key_parts = [calibration_tree.key_prefix]  # Use configurable prefix

    # Generate binary encoding in FIXED ORDER to match pre-trained GMMs
    binary_parts = []

    # Length (is_long)
    if 'short' in leaf_key:
        binary_parts.append('0')
    elif 'long' in leaf_key:
        binary_parts.append('1')
    else:
        binary_parts.append('2')

    # Disorder (is_disordered)
    if 'ordered' in leaf_key:
        binary_parts.append('0')
    elif 'disordered' in leaf_key:
        binary_parts.append('1')
    else:
        binary_parts.append('2')

    # Add PPI if present (though not in your current GMMs)
    if 'non_ppi' in leaf_key:
        binary_parts.append('0')
    elif 'ppi' in leaf_key:
        binary_parts.append('1')
    else:
        binary_parts.append('2')

    # Sulfur (sulfur_content)
    if 'non_sulfur' in leaf_key:
        binary_parts.append('0')
    elif 'sulfur' in leaf_key:
        binary_parts.append('1')
    else:
        binary_parts.append('2')

    key_parts.append(''.join(binary_parts))
    return '_'.join(key_parts)


def train_gmm(df: pd.DataFrame, score_col: str, label_col: str, n_components: int) -> tuple:
    """
    Train GMM models for benign and pathogenic subsets using GaussianMixture from sklearn.
    :param df: Input DataFrame
    :param score_col: The name of the score column
    :param label_col: The name of the label column
    :param n_components: The number of mixture components for the GMM
    :return: A tuple containing the trained benign and pathogenic GMM models
    """
    scores = df[score_col].to_numpy()
    labels = df[label_col].to_numpy().astype(int)

    X_ben, X_pat = scores[labels == 0].reshape(-1, 1), scores[labels == 1].reshape(-1, 1)
    if len(X_ben) == 0 or len(X_pat) == 0:
        raise ValueError("Both benign and pathogenic subsets must contain samples.")

    benign_gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=0).fit(X_ben)
    pathogenic_gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=0).fit(X_pat)

    return benign_gmm, pathogenic_gmm


def get_gmm_for_score_col(per_node_train_dfs: Dict[str, pd.DataFrame], score_col: str, calibration_tree: CalibrationTree, label_col: str, n_components: int) -> Dict[str, Tuple[GaussianMixture, GaussianMixture]]:
    """
    Train GMM models for each leaf combination in the calibration tree for a specific score column.
    :param per_node_train_dfs: A dictionary mapping node keys to their training DataFrames
    :param score_col: The name of the score column
    :param calibration_tree: The calibration tree
    :param label_col: The name of the label column
    :param n_components: The number of mixture components for the GMM
    :return: A dictionary mapping GMM keys to tuples of (benign_gmm, pathogenic_gmm)
    """

    if calibration_tree.root is None:
        raise ValueError("TreeSpec has not been built yet. Call tree_spec.build_tree() first.")

    all_gmms = {}

    updated_per_node_train_dfs = remove_outliers(per_node_train_dfs, score_col)

    # Train GMM for each leaf combination
    for leaf_key, subset in updated_per_node_train_dfs.items():

        if subset.empty:
            continue

        df_for_gmm = subset[[score_col, label_col]]

        # Generate key for this leaf combination
        gmm_key = generate_gmm_key_from_calibration_tree(calibration_tree, leaf_key)

        n_path = len(subset[subset[label_col] == 1])
        n_ben = len(subset[subset[label_col] == 0])

        leaf_name = key_to_name(calibration_tree, leaf_key)
        print(f"{gmm_key} ({leaf_name}): n_b: {n_ben} n_p: {n_path}")

        if n_path >= 5 and n_ben >= 5:
            try:
                benign_gmm, pathogenic_gmm = train_gmm(
                    df=df_for_gmm,
                    label_col=label_col,
                    score_col=score_col,
                    n_components=n_components
                )
            except Exception as e:
                print(f"Error training GMM for {gmm_key}: {e}")
                benign_gmm, pathogenic_gmm = None, None
        else:
            benign_gmm, pathogenic_gmm = None, None

        all_gmms[gmm_key] = (benign_gmm, pathogenic_gmm)

    return all_gmms


def train_gmms_for_tree_spec(per_node_train_dfs: Dict[str, pd.DataFrame], calibration_tree: CalibrationTree, score_cols: List[str], label_col: str, n_components: int):
    """
    Train GMM models for all score columns and leaf combinations in the calibration tree.
    :param per_node_train_dfs: A dictionary mapping node keys to their training DataFrames
    :param calibration_tree: The calibration tree
    :param score_cols: A list of score column names
    :param label_col: The name of the label column
    :param n_components: The number of mixture components for the GMM
    :return: A dictionary mapping score column names to dictionaries of GMM models
             (which map GMM keys to tuples of (benign_gmm, pathogenic_gmm))
    """
    if calibration_tree.root is None:
        raise ValueError("TreeSpec has not been built yet. Call tree_spec.build_tree() first.")

    # Train GMMs for each score column
    all_gmm_models = {}
    for score_col in score_cols:
        print(f"\nTraining GMMs for {score_col}")
        all_gmms = get_gmm_for_score_col(
            per_node_train_dfs=per_node_train_dfs,
            score_col=score_col,
            calibration_tree=calibration_tree,
            label_col=label_col,
            n_components=n_components
        )
        all_gmm_models[score_col] = all_gmms
    return all_gmm_models


def sample_from_gmm(gmm_model: tuple, n_samples: int, pathogenic_ratio: float) -> Dict[str, np.ndarray]:
    """
    Sample from the GMM model to generate synthetic scores and labels.
    The sampling is happening according to the specified pathogenic ratio.
    :param gmm_model: A tuple containing (benign_gmm, pathogenic_gmm)
    :param n_samples: Number of samples to generate
    :param pathogenic_ratio: The pathogenic ratio for each tree node
    :return: A dictionary with 'scores' and 'labels' as keys
    """
    if n_samples < 1:
        return {
            'scores': np.array([]),
            'labels': np.array([])
        }
    benign_gmm, pathogenic_gmm = gmm_model

    n_pathogenic = int(n_samples * pathogenic_ratio)
    n_benign = n_samples - n_pathogenic

    # Generate samples
    pathogenic_samples = pathogenic_gmm.sample(n_pathogenic)[0].flatten()
    benign_samples = benign_gmm.sample(n_benign)[0].flatten()

    # Combine
    scores = np.concatenate([pathogenic_samples, benign_samples])
    labels = np.concatenate([np.ones(n_pathogenic), np.zeros(n_benign)])

    return {
        'scores': scores,
        'labels': labels
    }

