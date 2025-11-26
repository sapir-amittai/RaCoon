import logging
import os
import numpy as np
import argparse

from consts import ESM_MAX_LENGTH, SCORE_COL, PPI_PREDICTIONS_PATH, WT_ENTROPY_COL_NAME, \
    PATHOGENIC, BENIGN, N_SAMPLES_TO_GENERATE, N_BINS, \
    GMM_CREATION_SAMPLE_SIZE, RANDOM_STATE, RATIO_SAMPLE_SIZE, PPI_PREDICTIONS_FILE_NAME
import utils
import pandas as pd

from gmm_utils import train_gmms_for_tree_spec, generate_gmm_key_from_tree_spec, sample_from_gmm
from tree_node import CalibrationTree, dim_sulfur, dim_ppi, dim_length, dim_fold, print_tree, \
    build_node_calibration_masks, key_to_name, build_leaf_masks


def build_train_test_per_nodes(
        df,
        calibration_tree,
        train_sample_size,
        random_state,
        ratio_sample_size):
    """
    Split data per node into train/test sets and calculate pathogenic ratios.

    Process:
    1. For each node, sample up to ratio_sample_size mutations to calculate pathogenic ratio
    2. Use these ratio samples for training (up to train_sample_size per class)
    3. If we need more training samples, take additional ones from remaining data
    4. Discard any unused ratio samples (they won't be in train OR test)
    5. Everything else goes to test

    Args:
        df: DataFrame with mutation data
        calibration_tree: TreeSpec object defining the tree structure
        train_sample_size: Number of samples per class for training (e.g., 100 means 100 path + 100 benign)
        random_state: Random state for reproducibility
        ratio_sample_size: Number of samples to use for calculating pathogenic ratio (default: 500)

    Returns:
        train_df: Training DataFrame
        test_df: Test DataFrame
        node_pathogenic_ratios: Dict mapping leaf_key to pathogenic ratio
    """
    # Build leaf masks FIRST (before splitting)
    leaf_masks = build_node_calibration_masks(df, calibration_tree)

    # Create empty train/test dataframes
    per_node_train_dfs = {}
    used_train_data = []
    node_pathogenic_ratios = {}

    # Split PER NODE
    for leaf_key, leaf_mask in leaf_masks.items():
        node_data = df[leaf_mask].copy()

        if len(node_data) == 0:
            print(f"Skipping node {leaf_key} - no data")
            continue

        # Split by class
        pathogenic_all = node_data[node_data['binary_label'] == PATHOGENIC]
        benign_all = node_data[node_data['binary_label'] == BENIGN]

        # =================================================================
        # STEP 1: Sample for ratio calculation
        # =================================================================
        # Sample up to ratio_sample_size from the full node data
        ratio_sample_size_actual = min(ratio_sample_size, len(node_data))
        ratio_sample = node_data.sample(n=ratio_sample_size_actual, random_state=random_state)

        # Calculate pathogenic ratio from these samples
        n_pathogenic_in_ratio_sample = ratio_sample['binary_label'].sum()
        pathogenic_ratio = n_pathogenic_in_ratio_sample / len(ratio_sample) if len(ratio_sample) > 0 else 0.5
        node_pathogenic_ratios[leaf_key] = pathogenic_ratio

        # Split ratio sample by class
        ratio_pathogenic = ratio_sample[ratio_sample['binary_label'] == PATHOGENIC]
        ratio_benign = ratio_sample[ratio_sample['binary_label'] == BENIGN]

        # =================================================================
        # STEP 2: Create training set from ratio samples + additional if needed
        # =================================================================

        # Start with ratio samples for training
        if len(ratio_pathogenic) >= train_sample_size:
            # We have enough pathogenic in ratio sample
            train_path = ratio_pathogenic.sample(n=train_sample_size, random_state=random_state)
        else:
            # Use all ratio pathogenic and get more from the rest
            train_path = ratio_pathogenic.copy()
            additional_path_needed = train_sample_size - len(ratio_pathogenic)

            # Get additional from pathogenic_all (excluding ratio samples)
            pathogenic_remaining = pathogenic_all.drop(ratio_sample.index, errors='ignore')

            if len(pathogenic_remaining) < additional_path_needed:
                additional_path_needed = len(pathogenic_remaining) // 2
            additional_path = pathogenic_remaining.sample(n=additional_path_needed, random_state=random_state)
            train_path = pd.concat([train_path, additional_path])

        # Same for benign
        if len(ratio_benign) >= train_sample_size:
            # We have enough benign in ratio sample
            train_ben = ratio_benign.sample(n=train_sample_size, random_state=random_state)
        else:
            # Use all ratio benign and get more from the rest
            train_ben = ratio_benign.copy()
            additional_benign_needed = train_sample_size - len(ratio_benign)

            # Get additional from benign_all (excluding ratio samples)
            benign_remaining = benign_all.drop(ratio_sample.index, errors='ignore')

            if len(benign_remaining) < additional_benign_needed:
                additional_benign_needed = len(benign_remaining) // 2
            additional_ben = benign_remaining.sample(n=additional_benign_needed, random_state=random_state)
            train_ben = pd.concat([train_ben, additional_ben])
        node_sed_train_data = pd.concat([train_path, train_ben, ratio_sample])
        node_sed_train_data = node_sed_train_data.drop_duplicates(subset=['protein_sequence', 'mutant'])
        used_train_data.append(node_sed_train_data)
        per_node_train_dfs[leaf_key] = pd.concat([train_path, train_ben])

        # =================================================================
        # STEP 3: Create test set (everything except train and unused ratio samples)
        # =================================================================

    node_sed_train_data_df = pd.concat(used_train_data).drop_duplicates(subset=['protein_sequence', 'mutant'])
    test_df = df.merge(
        node_sed_train_data_df[['protein_sequence', 'mutant']],
        on=['protein_sequence', 'mutant'],
        how='left',
        indicator=True
    ).query('_merge == "left_only"').drop(columns=['_merge'])

    return per_node_train_dfs, test_df, node_pathogenic_ratios


def get_calibrations(per_node_train_dfs, calibration_tree, score_col, gmm_models_by_score, node_pathogenic_ratios, ):
    calibrations = {}
    for leaf_key in per_node_train_dfs.keys():
        leaf_name = key_to_name(calibration_tree, leaf_key)

        gmm_key = generate_gmm_key_from_tree_spec(calibration_tree, leaf_key)
        gmm_data = gmm_models_by_score[score_col][gmm_key]
        benign_gmm, pathogenic_gmm = gmm_data

        # Sample from GMM to create synthetic data for calibration
        pathogenic_ratio_for_leaf = node_pathogenic_ratios[leaf_key]
        synthetic_data = sample_from_gmm(
            (benign_gmm, pathogenic_gmm),
            N_SAMPLES_TO_GENERATE,
            pathogenic_ratio_for_leaf,
        )

        # Create calibration from synthetic data
        calibration = utils.create_unified_calibration(
            synthetic_data=synthetic_data,
            score_col=score_col,
            n_bins=N_BINS,
            combination_strategy='synthetic_only'
        )

        if calibration is not None:
            calibrations[(leaf_key, score_col)] = {
                'calibration': calibration,
                'score_col': score_col,
                'leaf_key': leaf_key,
                'leaf_name': leaf_name
            }
    return calibrations


def apply_calibration_to_df(test_data, calibrated_col_name, tree_spec, calibrations):
    test_data[calibrated_col_name] = np.nan

    test_leaf_masks = build_leaf_masks(test_data, tree_spec)
    # Apply calibrations for each leaf group
    for (leaf_key, score_col), cal_info in calibrations.items():
        calibration = cal_info['calibration']
        leaf_name = cal_info['leaf_name']
        leaf_mask = test_leaf_masks[leaf_key]
        leaf_data = test_data[leaf_mask]
        if len(leaf_data) == 0:
            continue

        # Apply calibration to this group
        calibrated_scores = utils.apply_unified_calibration(
            test_data=leaf_data,
            score_col=score_col,
            calibration=calibration
        )

        # Store calibrated scores in the appropriate column
        test_data.loc[leaf_mask, calibrated_col_name] = calibrated_scores

        print(f"Applied calibration for {leaf_name} ({score_col}): {leaf_mask.sum()} mutations")
    return test_data


def racoon(
        df,
        tree_spec,
        gmm_creation_sample_size,
        random_state,
        ratio_sample_size,
        score_col,
        label_col
):
    per_node_train_dfs, test_df, node_pathogenic_ratios = build_train_test_per_nodes(
        df=df,
        calibration_tree=tree_spec,
        train_sample_size=gmm_creation_sample_size,
        random_state=random_state,
        ratio_sample_size=ratio_sample_size
    )

    gmm_models_by_score = train_gmms_for_tree_spec(
        per_node_train_dfs=per_node_train_dfs,
        tree_spec=tree_spec,
        score_cols=[score_col],
        label_col=label_col,
    )

    calibrations = get_calibrations(
        per_node_train_dfs=per_node_train_dfs,
        calibration_tree=tree_spec,
        score_col=score_col,
        gmm_models_by_score=gmm_models_by_score,
        node_pathogenic_ratios=node_pathogenic_ratios,
    )

    test_df_calibrated = apply_calibration_to_df(
        test_data=test_df,
        calibrated_col_name='our_calibrated_score',
        tree_spec=tree_spec,
        calibrations=calibrations,
    )

    utils.calculate_auc_and_thresholds(
        test_data=test_df_calibrated,
        score_col=SCORE_COL,
        label_col=label_col,
        calibrated_col_name='our_calibrated_score'
    )
    return test_df_calibrated


def create_parser():
    parser = argparse.ArgumentParser(
        description="Pipeline interface for protein-level analysis of missense variants in familial data"
    )

    parser.add_argument(
        "--df_path",
        type=str,
        default=None,
        required=False
    )

    parser.add_argument(
        "--save_df_path",
        type=str,
        default=None,
        required=False
    )

    return parser


def get_calibration_tree(df):
    calibration_tree = CalibrationTree(
        available_dims=[dim_fold(), dim_sulfur(), dim_ppi()],
        length_dim=dim_length(threshold=ESM_MAX_LENGTH),
        title_info="Dynamically constructed tree based on JSD",
    )
    calibration_tree.build_tree(
        df=df,
        score_col=WT_ENTROPY_COL_NAME,
        label_col='binary_label'
    )
    print_tree(calibration_tree.root, df=df)
    return calibration_tree


def main():
    parser = create_parser()
    args = parser.parse_args()

    log_file = os.environ.get('LOGGER_OUTPUT_PATH', 'racoon/racoon_default_log.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting the predictor...")
    print("Starting the predictor...")

    df = pd.read_parquet(args.df_path) if args.df_path.endswith('.parquet') else pd.read_csv(args.df_path)
    df = utils.clean_duplication_in_dataset(df)
    df = utils.add_raw_esm1b_score(df, SCORE_COL, WT_ENTROPY_COL_NAME)
    df = utils.add_is_ppi_mutation_column(
        df=df, zip_path=PPI_PREDICTIONS_PATH, json_filename_inside_zip=PPI_PREDICTIONS_FILE_NAME
    )
    calibration_tree = get_calibration_tree(df)
    df = racoon(
        df=df,
        tree_spec=calibration_tree,
        gmm_creation_sample_size=GMM_CREATION_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        ratio_sample_size=RATIO_SAMPLE_SIZE,
        score_col=SCORE_COL,
        label_col='binary_label'
    )
    if args.save_df_path is not None:
        os.makedirs(os.path.dirname(args.save_df_path), exist_ok=True)
        df.to_parquet(args.save_df_path) if args.save_df_path.endswith('.parquet') else df.to_csv(args.save_df_path)
        logger.info(f"Saved calibrated DataFrame to {args.save_df_path}")
        print(f"Saved calibrated DataFrame to {args.save_df_path}")


if __name__ == "__main__":
    main()
