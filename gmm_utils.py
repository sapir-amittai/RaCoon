import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from tree_node import CalibrationTree, key_to_name


def extreme_outliers_mad(data, threshold=4.25):
    data = np.array(data)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return np.array([], dtype=int)  # No variation = no outliers
    modified_z = 0.6745 * (data - median) / mad
    return np.where(np.abs(modified_z) > threshold)[0]  # returns indices of extreme outliers



def remove_outliers(per_node_train_dfs: pd.DataFrame, score_col, tree_spec):
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



def generate_gmm_key_from_tree_spec(calibration_tree: CalibrationTree, leaf_key) -> str:
    """Generate GMM key from TreeSpec and leaf_key."""
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


def train_gmm(df: pd.DataFrame, score_col: str, label_col: str, n_components: int):
    """
    Train GMM models for benign and pathogenic classes.

    Note: This function no longer returns pathogenic_ratio, n_path, n_ben since training data
    is now balanced. Use node_pathogenic_ratios from build_train_test_per_nodes instead.

    Args:
        df: DataFrame with score and label columns
        score_col: Name of score column
        label_col: Name of label column
        n_components: Number of GMM components

    Returns:
        benign_gmm: Trained GMM for benign class
        pathogenic_gmm: Trained GMM for pathogenic class
    """
    scores = df[score_col].to_numpy()
    labels = df[label_col].to_numpy().astype(int)

    X_ben, X_pat = scores[labels == 0].reshape(-1, 1), scores[labels == 1].reshape(-1, 1)
    if len(X_ben) == 0 or len(X_pat) == 0:
        raise ValueError("Both benign and pathogenic subsets must contain samples.")

    benign_gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=0).fit(X_ben)
    pathogenic_gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=0).fit(X_pat)

    return benign_gmm, pathogenic_gmm



def get_gmm_for_score_col(per_node_train_dfs, score_col, calibration_tree: CalibrationTree, label_col, n_components):
    """
    Train GMM models for each leaf combination defined by the dynamically built TreeSpec.

    NOTE: tree_spec.build_tree() must be called BEFORE this function.
    NOTE: This function no longer returns pathogenic_ratio, n_path, n_ben.
          GMMs are stored as (benign_gmm, pathogenic_gmm) only.
    """

    if calibration_tree.root is None:
        raise ValueError("TreeSpec has not been built yet. Call tree_spec.build_tree() first.")

    all_gmms = {}

    updated_per_node_train_dfs = remove_outliers(per_node_train_dfs, score_col, calibration_tree)

    # Train GMM for each leaf combination
    for leaf_key, subset in updated_per_node_train_dfs.items():
        # subset = filtered_data[leaf_mask].copy()

        if subset.empty:
            continue

        df_for_gmm = subset[[score_col, label_col]]

        # Generate key for this leaf combination
        gmm_key = generate_gmm_key_from_tree_spec(calibration_tree, leaf_key)

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

        # Store only the GMM models (no ratio, n_path, n_ben)
        all_gmms[gmm_key] = (benign_gmm, pathogenic_gmm)

    return all_gmms



def train_gmms_for_tree_spec(per_node_train_dfs, tree_spec, score_cols, label_col='binary_label', n_components=2):
    """
    Train GMMs for a dynamically built TreeSpec.

    Args:
        df: DataFrame with mutation data
        tree_spec: TreeSpec object (must have build_tree() called already)
        score_cols: List of score columns to train GMMs for
        label_col: Label column name
        n_components: Number of GMM components

    Returns:
        Dict of GMM models organized by score column
    """

    if tree_spec.root is None:
        raise ValueError("TreeSpec has not been built yet. Call tree_spec.build_tree() first.")

    # Preprocess data
    # df = preprocess_df(df, tree_spec)
    # Train GMMs for each score column
    all_gmm_models = {}
    for score_col in score_cols:
        print(f"\nTraining GMMs for {score_col}")
        all_gmms = get_gmm_for_score_col(
            per_node_train_dfs=per_node_train_dfs,
            score_col=score_col,
            calibration_tree=tree_spec,
            label_col=label_col,
            n_components=n_components
        )
        all_gmm_models[score_col] = all_gmms
    return all_gmm_models


def sample_from_gmm(gmm_model, n_samples, pathogenic_ratio):
    """Sample synthetic data from GMM model."""
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

