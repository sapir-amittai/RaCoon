import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from itertools import product
import ast


#  AUROC, ECE, MCE, J-SCORE, JSD COMPUTATIONS
def compute_auc_and_optimal_threshold(data: pd.DataFrame, score_col='score', label_col='label'):
    """
    Computes AUROC and the Youden-optimal classification threshold.
    :param data: DataFrame containing score and label columns.
    :param score_col: Score column name.
    :param label_col: Binary label column name.
    :return: AUROC and optimal threshold.
    """
    flip_flag = False
    y_true = data[label_col].values
    scores = data[score_col].values
    auc = roc_auc_score(y_true, scores)
    if auc < 0.5:
        scores = -scores
        auc = 1 - auc
        flip_flag = True
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    optimal_idx = j_scores.argmax()
    optimal_threshold = -thresholds[optimal_idx] if flip_flag else thresholds[optimal_idx]
    return auc, optimal_threshold

def compute_average_protein_auc(test_data: pd.DataFrame, score_col='score', label_col='label',
                                protein_col='protein_sequence', weighted_avg=False,
                                min_label_count=2):
    """
    This function evaluates AUC on a per-protein basis and provides
    an option to compute a weighted average based on the relative number of examples
    per protein.
    :param test_data: Dataframe containing the dataset to evaluate. It must include columns
        for scores, labels, and protein sequences as specified by `score_col`, `label_col`,
        and `protein_col`.
    :param score_col: Column name in the dataframe that contains the score predictions.
    :param label_col: Column name in the dataframe that contains the binary labels
    :param protein_col: Column name in the dataframe that specifies the protein sequences
    :param weighted_avg: Flag that, if set to True, computes a weighted average of AUCs
        based on the relative number of examples per protein. Default is False.
    :param min_label_count: Minimum number of labels required for a protein group to be
        included in the AUC calculation. Default is 2.
    :return: The average AUC across all proteins in the dataset. If `weighted_avg` is True,
        the function returns a tuple containing the weighted average AUC and the regular
        average AUC.
    :rtype: float | tuple[float, float]
    """
    # Global AUC check to determine if scores need to be flipped
    y_true = test_data[label_col].values
    scores = test_data[score_col].values
    global_auc = roc_auc_score(y_true, scores)

    if global_auc < 0.5:
        scores = -scores  # Flip all scores if needed
        test_data = test_data.copy()
        test_data[score_col] = scores

    # Compute per-protein AUC, skipping proteins with only one label class
    valid_aucs, relative_weight = [], []
    n_examples = 0
    for _, group in test_data.groupby(protein_col):
        y = group[label_col].values
        if len(set(y)) < 2 or len(y) < min_label_count:
            continue  # Skip proteins with only one label class or with too few labels
        n_examples += len(y)
        auc = roc_auc_score(y, group[score_col].values)
        valid_aucs.append(auc)
        relative_weight.append(len(y))
    if weighted_avg and n_examples > 0:
        return sum([score * (weight / n_examples) for score, weight in zip(valid_aucs, relative_weight)]), \
               sum(valid_aucs) / len(valid_aucs) if valid_aucs else float('nan')
    return sum(valid_aucs) / len(valid_aucs) if valid_aucs else float('nan')

def jsd_from_samples_hist(x, y, bins=100, range=None, base=2, eps=1e-12):
    """
    Computes the Jensen-Shannon Divergence (JSD) between two distributions represented
    as histograms.
    :param x: The first sample array used to construct a histogram.
    :param y: The second sample array used to construct a histogram.
    :param bins: The number of bins to use when constructing histograms.
        Defaults to 100.
    :param range: The range of histogram bins as a tuple (low, high). If None, the
        range is determined from the minimum and maximum values of the samples.
        Defaults to None.
    :param base: The logarithmic base used in the computation of divergence.
        Defaults to 2.
    :param eps: A small smoothing factor added to probabilities to avoid zeros
        in the histograms. Defaults to 1e-12.
    :return: A tuple containing the Jensen-Shannon distance and the squared
        Jensen-Shannon divergence.
    :rtype: tuple of (float, float)
    """
    # shared bins
    if range is None:
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        range = (lo, hi)
    Px, _ = np.histogram(x, bins=bins, range=range, density=False)
    Qy, _ = np.histogram(y, bins=bins, range=range, density=False)

    # convert to probabilities with smoothing to avoid zeros
    Px = Px.astype(float); Qy = Qy.astype(float)
    Px = (Px + eps) / (Px.sum() + eps * len(Px))
    Qy = (Qy + eps) / (Qy.sum() + eps * len(Qy))

    # scipy returns the **distance** (sqrt of JSD); square it to get divergence
    js_dist = jensenshannon(Px, Qy, base=base)
    js_div = js_dist**2
    return js_dist, js_div

def compute_ece_mce(y_true, y_score, n_bins=10, return_mce=True, min_samples_for_mce=50,
                    min_samples_for_ece=1):
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) based on
    a set of true labels and predicted probabilities. ECE measures the weighted average of
    the absolute difference between confidence and accuracy across bins, while MCE determines
    the largest calibration error among all bins.

    :param y_true: Array-like of shape (n_samples,). True binary labels of the dataset, where
                   values are either 0 or 1.
    :param y_score: Array-like of shape (n_samples,). Predicted probabilities for the positive
                    class for each sample.
    :param n_bins: Integer. Number of bins to partition the predicted probabilities. Default value is 10.
    :param return_mce: Boolean. If True, calculates the Maximum Calibration Error (MCE) alongside ECE.
    :param min_samples_for_mce: Integer. Minimum number of samples required in a bin for
                                considering it in the calculation of MCE. Default value is 50.
    :param min_samples_for_ece: Integer. Minimum number of samples required in a bin for
                                considering it in the calculation of ECE. Default value is 1.
    :return: Float, or Tuple[float, float]. The calculated ECE if `mce` is False, otherwise a tuple
             with the calculated ECE and MCE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    # Drop NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    N = y_true.size
    if N == 0:
        return np.nan

    # Bin edges over [0,1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # Assign each score to a bin index in [0, n_bins-1]
    bin_idx = np.digitize(y_score, bins, right=False) - 1
    # Clamp any edge cases (e.g., score==1.0 goes to last bin)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    ece, mce = 0.0, 0.0
    for b in range(n_bins):
        m = (bin_idx == b)
        n_b = int(m.sum())
        if n_b == 0 or n_b < min_samples_for_ece:
            continue
        # Empirical accuracy = mean of labels (positive rate)
        acc_b = float(y_true[m].mean())
        # Confidence = mean predicted probability
        conf_b = float(y_score[m].mean())
        err_b = abs(acc_b - conf_b)
        ece += (n_b / N) * abs(acc_b - conf_b)
        if err_b > mce and n_b>=min_samples_for_mce:
            mce = err_b

    return ece if not return_mce else ece,mce

#  CODE USED TO TRAIN LOGISTIC REGRESSION MODELS
#  sample logic
def train_test_split(data: pd.DataFrame, label_col: str, n_samples: int, k: int = 3,
                                  random_state: int = 42, model_col: str = None, unbalanced=False):
    """
    Splits the given dataset into training and testing sets k times with n_samples samples in each training set.
    Training set can either be balanced or unbalanced using the `unbalanced` flag.
    :param data: A pandas DataFrame containing the dataset to be split.
    :param label_col: The name of the column containing binary class labels 0|1|Na.
    :param n_samples: The desired total number of samples in each generated training set.
                      If this number exceeds the maximum possible balanced size, the maximum possible
                      value will be used instead.
    :param k: Number of splits or folds to generate. Defaults to 3 if not specified.
    :param random_state: Random seed to ensure reproducible splits across runs.
    :param model_col: An optional column name. If provided, rows where this column's value is null will
                      be excluded from the splitting process.
    :param unbalanced: A boolean flag that determines whether the split should be unbalanced.

    :return: Yields tuples of two DataFrames - (train_df, test_df), where `train_df` represents the
             training set for the current fold and `test_df` corresponds to the remaining data as
             the testing set.
    """
    if model_col is not None:
        data = data[data[model_col].notna()]
    df_pos = data[data[label_col] == 1].copy()
    df_neg = data[data[label_col] == 0].copy()
    max_possible = 2 * min(len(df_pos), len(df_neg))
    n_samples = n_samples if n_samples < max_possible else max_possible
    for i in range(k):
        if not unbalanced:
            pos_sample = df_pos.sample(n=n_samples // 2, random_state=random_state + i)
            neg_sample = df_neg.sample(n=n_samples // 2, random_state=random_state + 1000 + i)
            train_df = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=random_state + 2000 + i)
        if unbalanced:
            train_df = data.sample(n=n_samples, random_state=random_state + 3000 + i)
            train_df = train_df.sample(frac=1, random_state=random_state + 4000 + i)
        train_indices = train_df.index
        test_df = data.drop(train_indices).reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
        yield train_df, test_df

def extreme_outlier_removal_percentile(train_data: pd.DataFrame, model_col: str, label_col:str,
                                       lower_quantile=0.001, upper_quantile=0.999):
    """
    Removes extreme outliers from the data based on specified quantile ranges.
    :param train_data: Raw training data pandas DataFrame.
    :param model_col: Column name in the dataset upon which the outlier removal is performed.
    :param label_col: Column name in the dataset representing labels, which is preserved
        in the output.
    :param lower_quantile: Lower quantile threshold for filtering extreme outliers. Default value is 0.001.
    :param upper_quantile: Upper quantile threshold for filtering extreme outliers. Default value is 0.999.
    :return: Filtered dataset (pandas DataFrame) ready for LR training
        with extreme outliers removed based on the specified quantile thresholds.
    """
    model_train = train_data[train_data[model_col].notna()][[model_col, label_col]].copy()
    lower, upper = model_train[model_col].quantile([lower_quantile, upper_quantile])
    train_data = model_train[(model_train[model_col] >= lower) & (model_train[model_col] <= upper)]
    return train_data
# train LR logic
def train_logistic_regression(train_df: pd.DataFrame, score_col: str, label_col: str):
    """
    Trains a logistic regression model g(x) = sigmoid(c_1 * x + c_0) on a 1D feature.

    Args:
        train_df (pd.DataFrame): Training dataset.
        score_col (str): Column name of the input scores (feature x).
        label_col (str): Column name of the binary labels.

    Returns:
        model (LogisticRegression): Trained logistic regression model.
        coef_ (float): c_1 coefficient.
        intercept_ (float): c_0 intercept.
    """
    X = train_df[[score_col]].values  # shape (n_samples, 1)
    y = train_df[label_col].values  # shape (n_samples,)

    model = LogisticRegression(solver='lbfgs')
    model.fit(X, y)

    return model


#  CODE USED TO TRAIN GMMs
#  Outlier removal using MAD
def extreme_outliers_mad(data, threshold=4.25):
    data = np.array(data)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return np.array([], dtype=int)  # No variation = no outliers
    modified_z = 0.6745 * (data - median) / mad
    return np.where(np.abs(modified_z) > threshold)[0]  # returns indices of extreme outliers

def train_gmm(df: pd.DataFrame, score_col: str, label_col: str, n_components: int=2):
    scores = df[score_col].to_numpy()
    labels = df[label_col].to_numpy().astype(int)
    pathogenic_ratio = (labels == 1).sum() / labels.size if labels.size else 0.0
    X_ben, X_pat = scores[labels == 0].reshape(-1, 1), scores[labels == 1].reshape(-1, 1)
    if len(X_ben) == 0 or len(X_pat) == 0:
        raise ValueError("Both benign and pathogenic subsets must contain samples.")

    benign_gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=0).fit(X_ben)
    pathogenic_gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=0).fit(X_pat)
    return benign_gmm, pathogenic_gmm, pathogenic_ratio


#  HELPER FUNCTIONS FOR SPECIFIC FIGURES
#  helpers for figure 1A
def update_threshold_summary(results, attribute, thresholds, aucs, update_nan=False):
    """
    Updates threshold summary statistics for one attribute subset.
    :param results: Results accumulator dictionary.
    :param attribute: Attribute name prefix.
    :param thresholds: Bootstrap threshold values.
    :param aucs: Bootstrap AUROC values.
    :param update_nan: Whether to store missing values.
    :return: Updated results dictionary.
    """
    if update_nan:
        results[f'{attribute}_thr_mean'].append(np.nan)
        results[f'{attribute}_thr_std'].append(np.nan)
        results[f'{attribute}_auc_mean'].append(np.nan)
        results[f'{attribute}_auc_std'].append(np.nan)
    else:
        results[f'{attribute}_thr_mean'].append(np.array(thresholds).mean())
        results[f'{attribute}_thr_std'].append(np.array(thresholds).std())
        results[f'{attribute}_auc_mean'].append(np.array(aucs).mean())
        results[f'{attribute}_auc_std'].append(np.array(aucs).std())
    return results

# helpers for figure 1b - data for reliability histograms

def create_output_dict_1(n_bins, models):
    output_data = {'model': [model for model in models]}
    for i in range(n_bins):
        output_data[f'bin_{i}_all_ratios_mean'] = [[] for _ in models]
        output_data[f'bin_{i}_all_ratios_std'] = [[] for _ in models]
        output_data[f'bin_{i}_all_count_mean'] = [[] for _ in models]
        output_data[f'bin_{i}_att_ratios_mean'] = [[] for _ in models]
        output_data[f'bin_{i}_att_ratios_std'] = [[] for _ in models]
        output_data[f'bin_{i}_att_count_mean'] = [[] for _ in models]
    return output_data


def update_fold_results_1(results_dict, naive_ratios, att_ratios, naive_counts, att_counts, model_name,
                        model_idx, n_bins):
    idx = model_idx[model_name]
    for i in range(n_bins):
        results_dict[f'bin_{i}_all_ratios_mean'][idx].append(naive_ratios[i])
        results_dict[f'bin_{i}_all_count_mean'][idx].append(naive_counts[i])
        results_dict[f'bin_{i}_att_ratios_mean'][idx].append(att_ratios[i])
        results_dict[f'bin_{i}_att_count_mean'][idx].append(att_counts[i])
    return results_dict


def aggregate_results_over_folds_1(results_dict, n_bins, n_models):
    def safe_nanmean(arr):
        arr = np.asarray(arr)
        return np.nan if np.isnan(arr).all() else np.nanmean(arr)

    def safe_nanstd(arr):
        arr = np.asarray(arr)
        return np.nan if np.isnan(arr).all() else np.nanstd(arr)

    for i in range(n_bins):
        for j in range(n_models):
            naive_ratio_mean_j = safe_nanmean(results_dict[f'bin_{i}_all_ratios_mean'][j])
            naive_ratio_std_j = safe_nanstd(results_dict[f'bin_{i}_all_ratios_mean'][j])
            results_dict[f'bin_{i}_all_ratios_mean'][j] = naive_ratio_mean_j
            results_dict[f'bin_{i}_all_ratios_std'][j] = naive_ratio_std_j
            results_dict[f'bin_{i}_all_count_mean'][j] = safe_nanmean(
                results_dict[f'bin_{i}_all_count_mean'][j]
            )

            att_ratio_mean_j = safe_nanmean(results_dict[f'bin_{i}_att_ratios_mean'][j])
            att_ratio_std_j = safe_nanstd(results_dict[f'bin_{i}_att_ratios_mean'][j])
            results_dict[f'bin_{i}_att_ratios_mean'][j] = att_ratio_mean_j
            results_dict[f'bin_{i}_att_ratios_std'][j] = att_ratio_std_j
            results_dict[f'bin_{i}_att_count_mean'][j] = safe_nanmean(
                results_dict[f'bin_{i}_att_count_mean'][j]
            )
    return results_dict

def binned_pathogenic_to_benign_ratio(data, label_col, score_col, n_bins=10, min_bin_count=20):
    """
    Calculates the ratio of pathogenic to benign events per score bin, applying a cutoff for minimum count.

    :param data: pandas DataFrame containing the input data.
    :param label_col: Column name in the DataFrame indicating whether a data point is pathogenic (1) or benign (0).
    :param score_col: Column name in the DataFrame containing the scores used for binning.
    :param n_bins: Number of bins to divide the scores into. Default is 10.
    :param min_bin_count: minimum number of data points required in each bin. Default is 20.
    :return: A tuple of two lists:
             - List of pathogenic-to-total ratios for each bin (or NaN if count is insufficient).
             - List of total counts for each bin (or NaN if count is insufficient).
    """
    bins = np.linspace(0, 1, n_bins+1)  # 10 bins: [0.0, 0.1, ..., 1.0]
    bin_indices = np.digitize(data[score_col], bins, right=False) - 1
    bin_indices = bin_indices.clip(0, n_bins-1)  # ensure indices are in [0, 9]
    df = data.copy()
    df['bin'] = bin_indices
    ratios, counts = [], []
    for i in range(n_bins):
        bin_data = df[df['bin'] == i]
        count_pathogenic = (bin_data[label_col] == 1).sum()
        count_benign = (bin_data[label_col] == 0).sum()
        count = count_pathogenic + count_benign if count_pathogenic + count_benign >= min_bin_count else np.nan
        if count_pathogenic + count_benign >= min_bin_count:
            ratio = count_pathogenic / count
        else:
            ratio = np.nan
        ratios.append(ratio)
        counts.append(count)
    return ratios, counts

#  helpers for 2b (plot gmms)
def plot_llr_hist_with_thr(data, score_col, label_col, b_gmm=None, p_gmm=None, work_b_gmm=None, work_p_gmm=None,
                           legend=False, n_bins=100, title='', show_y=True, show_x=True,
                           return_fig=False, outpath='llr_dist.png'):
    fig, ax = plt.subplots(figsize=(4.15, 2.2)) if show_y else plt.subplots(figsize=(3.74, 2.2))
    fontsize = 7.5
    work = data[[score_col, label_col]].dropna().copy()
    scores = work[score_col].to_numpy()
    xmin, xmax = float(np.min(scores)), float(np.max(scores))
    _, thr = compute_auc_and_optimal_threshold(work, score_col, label_col)
    ax.vlines(x=thr, ymin=-0.01, ymax=0.35 - 0.025, colors='black', linestyles='--', alpha=0.25, linewidth=1)
    ax.text(thr, 0.327, f"{thr:.2f}", ha='center', va='bottom', fontsize=fontsize)
    path_naive_color = '#956CB4'  # Purple light
    path_work_color = '#7C5B98'  # Purple dark
    ben_naive_color = '#4878D0'  # Blue light
    ben_work_color = '#345695'  # Blue dark
    # compute draw gmm_boundaries
    def gmm_pdf_1d(gmm, xgrid):
        means = gmm.means_.ravel()
        vars_ = gmm.covariances_.ravel()
        weights = gmm.weights_.ravel()
        pdf = np.zeros_like(xgrid, dtype=float)
        for w, m, v in zip(weights, means, vars_):
            coef = 1.0 / np.sqrt(2.0 * np.pi * v)
            pdf += w * coef * np.exp(-0.5 * ((xgrid - m) ** 2) / v)
        return pdf

    if b_gmm is not None and p_gmm is not None:
        x = np.linspace(xmin, xmax, 1000)
        pdf_b = gmm_pdf_1d(b_gmm, x)
        pdf_p = gmm_pdf_1d(p_gmm, x)
        plt.plot(x, pdf_b, linewidth=1, color=ben_naive_color, linestyle='-', label='Benign GMM (pdf)', alpha=0.9)
        plt.plot(x, pdf_p, linewidth=1, color=path_naive_color, linestyle='-', label='Pathogenic GMM (pdf)', alpha=0.9)
    if work_b_gmm is not None and work_p_gmm is not None:
        x = np.linspace(xmin, xmax, 1000)
        pdf_b_w = gmm_pdf_1d(work_b_gmm, x)
        pdf_p_w = gmm_pdf_1d(work_p_gmm, x)
        plt.plot(x, pdf_b_w, linewidth=1, color=ben_work_color, linestyle=(0, (5, 2)), label='Benign GMM (pdf)')
        plt.plot(x, pdf_p_w, linewidth=1, color=path_work_color, linestyle=(0, (5, 2)), label='Pathogenic GMM (pdf)')
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('black')
    ax.set_ylim(-0.01, 0.39)
    ax.set_xlim(-22.0, 0.05)
    if show_x:
        ax.set_xticks([-20, -15, -10, -5, 0])
        ax.set_xlabel('LLR Score', fontsize=fontsize)
    else:
        ax.set_xticks([])
    if show_y:
        ax.set_ylabel('Density', fontsize=fontsize, labelpad=2)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    else:
        ax.set_ylabel('')
        ax.set_yticks([])
    ax.tick_params(axis='both', labelsize=fontsize)
    if legend:
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color=path_work_color, linewidth=1, linestyle='-', label="Subgroups Pathogenic"),
                plt.Line2D([0], [0], color=ben_work_color, linewidth=1, linestyle='-', label="Subgroups Benign"),
                plt.Line2D([0], [0], color='none', label=""),
                plt.Line2D([0], [0], color=path_naive_color, linewidth=1, linestyle='--', label="Naive Pathogenic"),
                plt.Line2D([0], [0], color=ben_naive_color, linewidth=1, linestyle='--', label="Naive Benign"),
            ],
            loc='upper left',
            frameon=False,
            handlelength=2,
            handletextpad=0.5,
            labelspacing=0.3, fontsize=fontsize
        )
    ax.set_title(title, fontsize=fontsize)
    plt.tight_layout()
    if return_fig:
        return fig
    else:
        plt.savefig(outpath, dpi=600, transparent=True)

#  helpers for figure 3
def per_attribute_train_test_n_samples(data: pd.DataFrame, label_col: str, n_samples: int, k: int = 100,
                                       random_state: int = 42, subgroups: list = [], model_col: str = None,
                                       min_samples_required=None, cutomize_selection: list = [], test_data=None):
    """
    Draws k times n values per binary selection of attributes supplied
    i.e. 2 sets per attribute

    Yields:
       dict{att_pos: pd.DataFrame, att_neg: pd.DataFrame...}, test_df (remaining data).
    """
    if model_col is not None:
        data = data[data[model_col].notna()]
        if test_data is not None:
            test_data = test_data[test_data[model_col].notna()]

    min_samples_required = n_samples * 2 if min_samples_required is None else min_samples_required
    sel_vals = [[0], [1]]  #  positive and negative subgroup selections
    if len(subgroups) == 1:
        att = subgroups[0]
        data = data[data[att].notna()]
        for i in range(k):
            train_samples, test_sample = {}, data.copy()
            selction_iter = cutomize_selection if cutomize_selection else sel_vals
            for sel in selction_iter:
                work_data = data[(data[att].isin(sel))].copy()
                assert len(work_data) > min_samples_required, 'selection smaller then n_samples'
                train_sample = work_data.sample(n=n_samples, random_state=random_state + i)
                train_sample_indices = train_sample.index
                test_sample = test_sample.drop(train_sample_indices) if test_data is None else \
                    test_data[test_data[att].notna()].copy().reset_index(drop=True)
                train_samples[f'{att}__{sel}'] = train_sample
            yield train_samples, test_sample

    elif len(subgroups) == 2:
        att1, att2 = subgroups[0], subgroups[1]
        data = data[(data[att1].notna()) & (data[att2].notna())]
        for i in range(k):
            train_samples, test_sample = {}, data.copy()
            selction_iter = cutomize_selection if cutomize_selection else list(
                product(sel_vals, repeat=len(subgroups)))
            for sel_1, sel_2 in selction_iter:
                work_data = data[(data[att1].isin(sel_1)) & (data[att2].isin(sel_2))].copy()
                assert len(work_data) > min_samples_required, 'selection smaller then n_samples'
                train_sample = work_data.sample(n=n_samples, random_state=random_state + i)
                train_sample_indices = train_sample.index
                test_sample = test_sample.drop(train_sample_indices) if test_data is None else \
                    test_data[(test_data[att1].notna()) & (test_data[att2].notna())].copy().reset_index(drop=True)
                train_samples[f'{att1}__{sel_1}:{att2}__{sel_2}'] = train_sample
            yield train_samples, test_sample

def logistic_calibration_from_train_dict(test_data, train_dict, score_col: str, label_col: str, num_att=None):
    """
    The function applies segmentation based on residue attributes and performs
    logistic calibration for each relevant subset.

    :param test_data: The test dataset. This is a DataFrame that will be differentially calibrated
    :param train_dict: A dictionary where each key is an attribute or a combination of
        attributes that defines a subset in test data. The value for each key is a
        DataFrame used to train the logistic regression model for calibration.
    :param score_col: The name of the score column in both test and train datasets
    :param label_col: The name of the label column in the train datasets, used as
    :param num_att: Denotes the number of attributes that define a segment. If set to 2,
        two attributes will be used for segmentation. Defaults to None (uses a single
        attribute for segmentation).
    :return: The test dataset with the `score_col` calibrated based on the logistic
        regression models trained on corresponding subsets defined in `train_dict`.
    :rtype: pandas.DataFrame the diff
    """
    test_data = test_data.copy()
    for att, train_data in train_dict.items():
        train_data = train_data.dropna(subset=[score_col])
        lower, upper = train_data[score_col].quantile([0.005, 0.995])
        train_data = train_data[(train_data[score_col] >= lower) & (train_data[score_col] <= upper)]
        calibration_model = train_logistic_regression(train_df=train_data,
                                                      score_col=score_col,
                                                      label_col=label_col)
        att_list = att.split(':')
        att1, sel1 = att_list[0].split('__')[0], ast.literal_eval(att_list[0].split('__')[1])
        if num_att == 2:
            att2, sel2 = att_list[1].split('__')[0], ast.literal_eval(att_list[1].split('__')[1])
            region_mask = (test_data[att1].isin(sel1)) & (test_data[att2].isin(sel2))
        else:
            region_mask = test_data[att1].isin(sel1)

        if region_mask.any():
            test_data.loc[region_mask, score_col] = calibration_model.predict_proba(
                test_data.loc[region_mask, [score_col]]
            )[:, 1]

    return test_data

def create_output_dict_2(models:list):
    output_data = {'model': [model for model in models]}
    output_data[f'naive_auc_mean'] = [[] for _ in models]
    output_data[f'naive_per_prot_auc_mean'] = [[] for _ in models]
    output_data[f'naive_auc_std'] = [[] for _ in models]
    output_data[f'naive_per_prot_auc_std'] = [[] for _ in models]
    output_data[f'calibrated_auc_mean'] = [[] for _ in models]
    output_data[f'calibrated_per_prot_auc_mean'] = [[] for _ in models]
    output_data[f'calibrated_auc_std'] = [[] for _ in models]
    output_data[f'calibrated_per_prot_auc_std'] = [[] for _ in models]
    return output_data


def update_output_dict_2(results_dict, model, naive_auc, calibrated_auc, naive_prot_auc, calibrated_prot_auc):
    idx = results_dict['model'].index(model)
    results_dict['naive_auc_mean'][idx].append(naive_auc)
    results_dict['naive_per_prot_auc_mean'][idx].append(naive_prot_auc)
    results_dict['calibrated_auc_mean'][idx].append(calibrated_auc)
    results_dict['calibrated_per_prot_auc_mean'][idx].append(calibrated_prot_auc)
    return results_dict


def aggregate_results_over_folds_2(results_dict):
    def safe_nanmean(arr):
        arr = np.asarray(arr)
        return np.nan if np.isnan(arr).all() else np.nanmean(arr)

    def safe_nanstd(arr):
        arr = np.asarray(arr)
        return np.nan if np.isnan(arr).all() else np.nanstd(arr)

    for i in range(len(results_dict['model'])):
        for key, values in results_dict.items():
            if key == 'model' or '_std' in key:
                continue
            else:
                respective_std_key = key.replace('_mean', '_std')
                results_dict[respective_std_key][i] = safe_nanstd(results_dict[key][i])
                results_dict[key][i] = safe_nanmean(results_dict[key][i])

    return results_dict
