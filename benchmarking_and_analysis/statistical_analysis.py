from constants import *
from utils_ import *
import pandas as pd
import numpy as np
from tqdm import tqdm

# DATASETS

CLINVAR_BM = ''
CLINVAR_HQ = ''
PROTEIN_GYM = ''

# Data Preprocessing
def compute_residue_level_attributes(data: pd.DataFrame):
    """
    Computes binary residue-level attributes for amino acids in the given DataFrame.
    :param data: Pandas DataFrame containing a 'mutant' column in format [AA][idx][AA]
    :return: Updated DataFrame with boolean columns for residue-level attributes.
    """
    data['is_polar'] = data['mutant'].str.match(r'^[STCYNQG]', na=False)
    data['is_polar'] = data['is_polar'].astype(int)
    data['is_hydrophobic'] = data['mutant'].str.match(r'^[AVILMFYW]', na=False).astype(int)
    data['is_charged'] = data['mutant'].str.match(r'^[RHKDE]', na=False).astype(int)
    data['is_aromatic'] = data['mutant'].str.match(r'^[FWYH]', na=False).astype(int)
    data['is_acidic'] = data['mutant'].str.match(r'^[DE]', na=False).astype(int)
    data['is_basic'] = data['mutant'].str.match(r'^[RHK]', na=False).astype(int)
    data['is_polar_uncharged'] = data['mutant'].str.match(r'^[STNQCY]', na=False).astype(int)
    data['is_small'] = data['mutant'].str.match(r'^[GASTP]', na=False).astype(int)
    data['is_sulfur'] = data['mutant'].str.match(r'^[CM]', na=False).astype(int)
    data['is_proline_or_glycine'] = data['mutant'].str.match(r'^[PG]', na=False).astype(int)
    data['is_aliphatic'] = data['mutant'].str.match(r'^[VILMA]', na=False).astype(int)
    data['is_helix_breaker'] = data['mutant'].str.contains(r'^[GP]', na=False).astype(int)
    data['is_beta_branched'] = data['mutant'].str.contains(r'^[IVT]', na=False).astype(int)
    data['first_met'] = data['mutant'].str.match(rf'^[{AA}]1[{AA}]$')
    return data

#  MAD EXTREME OUTLIER REMOVAL AND MIN-MAX NORMALIZATION
def extreme_outliers_mad(data, threshold=4.25):
    """NaN-safe MAD outlier detection; returns indices (in the input array) of outliers."""
    data = np.array(data, dtype=float)
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))
    if mad == 0 or np.isnan(mad):
        return np.array([], dtype=int)  # no variation or only NaNs
    modified_z = 0.6745 * (data - median) / mad
    return np.where(np.abs(modified_z) > threshold)[0]

def normalize_min_max(data:pd.Series, remove_outliers=True, outliers_std_thr=4.25, flip=False):
    scores = data.values.astype(float)
    if remove_outliers:
        outliers_idxs = extreme_outliers_mad(data, threshold=outliers_std_thr)
    inlier_mask = ~np.isnan(scores)
    if remove_outliers:
        if outliers_idxs.size > 0:
            inlier_mask[outliers_idxs] = False
    inliers = scores[inlier_mask]
    vmin, vmax = np.nanmin(inliers), np.nanmax(inliers)
    normalized_scores = (data - vmin) / (vmax - vmin)
    normalized_scores = normalized_scores.clip(lower=0.0, upper=1.0)
    return 1.0 - normalized_scores if flip else normalized_scores

# Analysis for figure-1a
# optimal classification thresholds
def compute_optimal_classification_thresholds(benchmark_path=CLINVAR_BM, outpath='thr_changes_RESULTS.csv',
                                              n_bootstraps=1000):
    """
    Computes bootstrap optimal thresholds across ClinVar attribute subsets.
    :param benchmark_path: Input benchmark table path.
    :param outpath: Output CSV path.
    :param n_bootstraps: Number of bootstrap samples per subset.
    :return: Threshold summary table.
    """
    benchmark_data = pd.read_csv(benchmark_path)
    subset_order = [
        ('naive', None),
        ('idr', 'is_disordered'),
        ('low_homology', 'low_homologs'),
        ('polar', 'is_polar'),
        ('hydrophobic', 'is_hydrophobic'),
        ('charged', 'is_charged'),
        ('ppi', 'is_ppi'),
        ('ordered', 'is_disordered'),
        ('sulfur', 'is_sulfur'),
        ('aromatic', 'is_aromatic'),
    ]

    results = {'model': [], 'supervision_type': []}
    for attribute, _ in subset_order:
        for metric in ['thr_mean', 'thr_std', 'auc_mean', 'auc_std']:
            results[f'{attribute}_{metric}'] = []

    for score, supervision_type in MODELS_FOR_1A.items():
        work_data = benchmark_data[benchmark_data[score].notna()].copy()
        subsets = {
            'naive': work_data[[score, 'label']].copy(),
            'idr': work_data[work_data['is_disordered'] == 1][[score, 'label']].copy(),
            'low_homology': work_data[work_data['low_homologs'] == 1][[score, 'label']].copy(),
            'polar': work_data[work_data['is_polar'] == 1][[score, 'label']].copy(),
            'hydrophobic': work_data[work_data['is_hydrophobic'] == 1][[score, 'label']].copy(),
            'charged': work_data[work_data['is_charged'] == 1][[score, 'label']].copy(),
            'ppi': work_data[work_data['is_ppi'] == 1][[score, 'label']].copy(),
            'ordered': work_data[work_data['is_disordered'] == 0][[score, 'label']].copy(),
            'sulfur': work_data[work_data['is_sulfur'] == 1][[score, 'label']].copy(),
            'aromatic': work_data[work_data['is_aromatic'] == 1][[score, 'label']].copy(),
        }

        results['model'].append(score)
        results['supervision_type'].append(supervision_type)
        for attribute, _ in subset_order:
            subset = subsets[attribute]
            n_ben, n_path = len(subset[subset['label'] == 0]), len(subset[subset['label'] == 1])
            if n_path < 50 or n_ben < 50 or n_path + n_ben < 100:
                results = update_threshold_summary(results, attribute, [], [], update_nan=True)
                continue
            thresholds, aucs = [], []
            for _ in range(n_bootstraps):
                sample = subset.sample(n=len(subset), replace=True).reset_index(drop=True)
                auc, threshold = compute_auc_and_optimal_threshold(sample, score_col=score, label_col='label')
                thresholds.append(threshold)
                aucs.append(auc)
            results = update_threshold_summary(results, attribute, thresholds, aucs)

    results = pd.DataFrame(results)
    results.to_csv(outpath)
    return results

#  Analysis for figure 1b
#  NOTE ECE,MCE Calculation in utils_
def compute_subgroup_specific_miscalibration(data: pd.DataFrame, label_col: str,
                                             subgroups_cols: list=ALL_RESIDUE_SUBGROUPS, models: list=MODELS_FOR_1B,
                                             n_bootstraps=100,
                                             outpath='subgroup_specific_miscalibration_aggregated.csv'):
    """
    Computes subgroup-specific calibration histograms over bootstrap train/test splits.

    :param data: Input dataframe containing labels, model score columns, and binary subgroup columns.
    :param label_col: Binary label column name (0/1).
    :param subgroups_cols: Binary subgroup columns to evaluate; one output row is produced per subgroup-model pair.
    :param models: Model score columns to calibrate and evaluate.
    :param n_bootstraps: Number of bootstrap train/test splits.
    :param outpath: Output CSV path for the aggregated results table.
    :return: Aggregated dataframe in wide format with columns:
        `subgroup`, `model`, and for each bin `i` in 0..9:
        `bin_i_all_ratios_mean`, `bin_i_all_ratios_std`, `bin_i_all_count_mean`,
        `bin_i_att_ratios_mean`, `bin_i_att_ratios_std`, `bin_i_att_count_mean`.
        The `all_*` columns are the global calibrated test-set values and are repeated across subgroup rows
        for the same model; the `att_*` columns are subgroup-specific.
    """
    n_bins = 10
    model_idx = {model: i for i, model in enumerate(models)}
    results = {subgroup: create_output_dict_1(n_bins=n_bins, models=models) for subgroup in subgroups_cols}
    for train, test in train_test_split(data=data, label_col=label_col, n_samples=6000, k=n_bootstraps):
        for model_score_col in models:
            train_data = extreme_outlier_removal_percentile(train_data=train, model_col=model_score_col,
                                                            label_col=label_col)
            calibrated_model = train_logistic_regression(train_df=train_data, score_col=model_score_col,
                                                         label_col=label_col)
            model_test = test[test[model_score_col].notna()].copy()
            model_test.loc[:, model_score_col] = calibrated_model.predict_proba(model_test[[model_score_col]])[:, 1]
            global_ratios, global_counts = binned_pathogenic_to_benign_ratio(model_test, label_col, model_score_col)
            for subgroup in subgroups_cols:
                subgroup_data = model_test[model_test[subgroup] == 1][[model_score_col, label_col]].copy()
                subgroup_ratios, subgroup_counts = binned_pathogenic_to_benign_ratio(subgroup_data, label_col,
                                                                                     model_score_col)
                results[subgroup] = update_fold_results_1(results[subgroup], global_ratios, subgroup_ratios,
                                                        global_counts, subgroup_counts, model_score_col,
                                                        model_idx=model_idx, n_bins=n_bins)
    aggregate_results = []
    for subgroup in subgroups_cols:
        subgroup_results = aggregate_results_over_folds_1(results[subgroup], n_bins=n_bins, n_models=len(models))
        subgroup_df = pd.DataFrame(subgroup_results)
        subgroup_df.insert(0, 'subgroup', subgroup)
        aggregate_results.append(subgroup_df)
    aggregate_df = pd.concat(aggregate_results, ignore_index=True)
    aggregate_df.to_csv(outpath, index=False)
    return aggregate_df

#  Analysis for figure 2a - See Extended Table S2
#  Analysis for figure 2b, 2c
def plot_score_distributions(data: pd.DataFrame, score_column: str, label_column: str, subgroup_col: str,
                                 outpath: str):
    """
    Plots model score distributions comparing the global data with a subgroup of interest
    using Gaussian Mixture Models (GMM). This function trains separate GMMs for the global data and the subgroup
    while maintaining the same pathogenic-to-benign ratio.

    NOTE: no outlier removal is performed. Consider MAD or percentile outlier removal.
    :param data: The input data as a pandas DataFrame containing relevant columns for scores, labels, and subgroups.
    :param score_column: The column name representing the model scores in the input DataFrame.
    :param label_column: The column name specifying binary labels for benign (0) and pathogenic (1).
    :param subgroup_col: The column name identifying the subgroup indicator (binary values: 1 for subgroup).
    :param outpath: The file path where the plot will be saved.
    :return: None. The function saves the plot to the specified path.
    :rtype: None
    """
    working_data = data[[score_column, label_column, subgroup_col]].copy()
    global_data_for_gmm = working_data[working_data[score_column].notna()]
    n_ben_global, n_path_global = (len(global_data_for_gmm[global_data_for_gmm[label_column] == 0]),
                                   len(global_data_for_gmm[global_data_for_gmm[label_column] == 1]))
    global_benign_gmm, global_pathogenic_gmm, _ = (
        train_gmm(df=global_data_for_gmm, score_col=score_column, label_col=label_column))
    subgroup_data_for_gmm = working_data[working_data[subgroup_col] == 1][[score_column, label_column]].copy()
    n_ben_subgroup, n_path_subgroup = (len(subgroup_data_for_gmm[subgroup_data_for_gmm[label_column] == 0]),
                                       len(subgroup_data_for_gmm[subgroup_data_for_gmm[label_column] == 1]))
    subgroup_path_data, subgroup_ben_data = (subgroup_data_for_gmm[subgroup_data_for_gmm[label_column] == 1].copy(),
                                             subgroup_data_for_gmm[subgroup_data_for_gmm[label_column] == 0].copy())
    # train gmms with the same pathogenic to benign ratio as the global gmm
    scale_subgroup = min(n_path_subgroup / n_path_global, n_ben_subgroup / n_ben_global)
    n_path_sample, n_ben_sample = int(n_path_global * scale_subgroup), int(n_ben_global * scale_subgroup)
    subgroup_path_sample, subgroup_ben_samples = subgroup_path_data.sample(n=n_path_sample, random_state=10), \
        subgroup_ben_data.sample(n=n_ben_sample, random_state=10)
    unified_subgroup_scaled_data = (pd.concat([subgroup_path_sample, subgroup_ben_samples]).
                                    sample(frac=1, random_state=10).reset_index(drop=True))
    unified_subgroup_scaled_data = unified_subgroup_scaled_data[unified_subgroup_scaled_data[score_column].notna()]
    subgroup_benign_gmm, subgroup_pathogenic_gmm, _ = train_gmm(df=unified_subgroup_scaled_data,
                                                        label_col=label_column, score_col=score_column, n_components=2)
    title = f'{subgroup_col} (P={n_path_sample}, B={n_ben_sample})'
    plot_llr_hist_with_thr(data, score_col=score_column, label_col=label_column,
                           p_gmm=global_pathogenic_gmm, b_gmm=global_benign_gmm,
                           work_b_gmm=subgroup_benign_gmm, work_p_gmm=subgroup_pathogenic_gmm,
                           n_bins=100, outpath=outpath, title=title, legend=True, show_y=True)

#  Analysis of differential score mapping - figure 3, 4
#  For figure 4(d,e) can run the same code with different score columns
def compute_auc_for_differential_calibration(data: pd.DataFrame, label_col: str, protein_seq_col: str,
                                             unsupervised_models: list=MODELS_FOR_3_UNSUPERVISED,
                                             supervised_models: list=MODELS_FOR_3_SUPERVISED, n_bootstraps=100,
                                             n_samples=500):
    """
    Computes the Area Under the Curve (AUC) for differential calibration of supervised
    and unsupervised models across specified residue subgroups. The function performs
    bootstrapping and evaluates the difference in AUC before and after calibration
    for each subgroup.
    NOTE: computing protein-level AUCs is computationally heavy and can be omitted or downsampled to save time.
    :param data: Dataset to perform the analysis on.
    :param label_col: Name of the column containing the labels.
    :param protein_seq_col: Name of the column containing the protein sequence identifiers.
    :param unsupervised_models: List of unsupervised models to analyze.
    :param supervised_models: List of supervised models to analyze.
    :param n_bootstraps: Number of bootstrapping iterations to perform.
    :param n_samples: Number of samples to use in each bootstrapping iteration.
    :return: This function does not return a direct value but writes aggregated results
        for each residue subgroup and supervision type (supervised/unsupervised) to CSV files.
    :rtype: None
    """
    for supervision, models in zip(['unsupervised', 'supervised'], [unsupervised_models, supervised_models]):
        results = {att: create_output_dict_2(models=models) for att in RESIDUE_SUBGROUPS_FOR_3}
        for att in RESIDUE_SUBGROUPS_FOR_3:
            subgroups = att.split(':')  # use ':' for multiple attributes
            custom_selection = CUSTOM_SELECTION[att]
            for model in models:
                iter = 0
                for train_dict, test in tqdm(per_attribute_train_test_n_samples(data=data, label_col=label_col,
                                                                                n_samples=n_samples, k=n_bootstraps,
                                                                                subgroups=subgroups, model_col=model,
                                                                                cutomize_selection=custom_selection),
                                             desc=f'{model}_{att}', total=n_bootstraps):
                    naive_test = test.copy()
                    calibrated_test = test.copy()
                    calibrated_test = logistic_calibration_from_train_dict(test, train_dict, score_col=model,
                                                                           label_col=label_col, num_att=len(subgroups))
                    naive_auc, _ = compute_auc_and_optimal_threshold(naive_test, score_col=model, label_col=label_col)
                    calibrated_auc, _ = compute_auc_and_optimal_threshold(calibrated_test, score_col=model,
                                                                          label_col=label_col)
                    # NOTE This is computationally heavy
                    naive_prot_auc = compute_average_protein_auc(naive_test, score_col=model, label_col=label_col,
                                                                 protein_col=protein_seq_col)
                    calibrated_prot_auc = compute_average_protein_auc(calibrated_test, score_col=model,
                                                                      label_col=label_col,
                                                                      protein_col=protein_seq_col)

                    results[att] = update_output_dict_2(results[att], model, naive_auc, calibrated_auc, naive_prot_auc,
                                                      calibrated_prot_auc)
                    iter += 1
        for att in RESIDUE_SUBGROUPS_FOR_3:
            agg_results = aggregate_results_over_folds_2(results[att])
            if ':' in att:
                name = att.replace(':', '_')
            else:
                name = att
            pd.DataFrame(data=agg_results).to_csv(
                f'delta_aucs_{supervision}_{name}_boot_{n_bootstraps}_samples_{n_samples}.csv')

# JSD Calculation
def jsd_between_subgroups(data: pd.DataFrame, label_col: str, score_col: str, score_range: list=[0,1],
                          subgroups: list=ALL_RESIDUE_SUBGROUPS):
    results = {'subgroup': [], 'jsd': []}
    for subgroup in subgroups:
        subgroup_mask = data[subgroup] == 1
        positive_selection_benign_scores = data[(subgroup_mask) & (data[label_col] == 0)][score_col]
        positive_selection_pathogenic_scores = data[(subgroup_mask) & (data[label_col] == 1)][score_col]
        negative_selection_benign_scores = data[(~subgroup_mask) & (data[label_col] == 0)][score_col]
        negative_selection_pathogenic_scores = data[(~subgroup_mask) & (data[label_col] == 1)][score_col]
        _, ben_js_div = jsd_from_samples_hist(positive_selection_benign_scores, negative_selection_benign_scores,
                                              range=score_range)
        _, path_js_div = jsd_from_samples_hist(positive_selection_pathogenic_scores, negative_selection_pathogenic_scores,
                                               range=score_range)
        results['subgroup'].append(subgroup)
        results['jsd'].append(path_js_div + ben_js_div)
    return pd.DataFrame(data=results)
