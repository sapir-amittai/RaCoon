import json
import zipfile
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import metapredict as meta

from sklearn.metrics import auc, roc_curve

from consts import *
from data_classes import AAMut
from mutation_record import MutationRecord


def esm_setup(model_name=ESM1B_MODEL, device=DEVICE) -> tuple:
    """
    Load the ESM model and alphabet.
    :param model_name: The name of the ESM1b model to load.
    :param device: The device to load the model onto.
    :return: Tuple of (model, alphabet)
    """
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    model = model.to(device)
    print(f"model loaded on {device}")
    return model, alphabet


def process_mutation_name(mutation: str, offset: int) -> AAMut:
    """
    Process mutation name into AAMut dataclass.
    :param mutation: Mutation string (e.g., 'A123C')
    :param offset: Offset to apply to mutation index
    :return: AAMut dataclass
    """
    return AAMut(
        wt_aa=mutation[0],
        mut_idx=int(mutation[1:-1]) - offset,
        change_aa=mutation[-1],
    )


def identify_idrs(disorder_values: list, threshold: float, min_length: int=3) -> List[Tuple[int, int]]:
    """
    Get a list of intrinsically disordered regions (IDRs) from disorder scores.
    :param disorder_values: List of disorder scores for each residue
    :param threshold: A threshold above which a residue is considered disordered
    :param min_length: Minimum length of an IDR
    :return: List of (start, end) tuples for IDRs (1-indexed)
    """
    idrs = []
    in_idr = False
    start = 0

    for i, score in enumerate(disorder_values):
        if score >= threshold and not in_idr:
            # Start of a new IDR
            in_idr = True
            start = i + 1  # 1-indexed position
        elif score < threshold and in_idr:
            # End of current IDR
            end = i  # End position is the last disordered residue
            if end - start + 1 >= min_length:
                idrs.append((start, end))
            in_idr = False

    # Check if we ended while still in an IDR
    if in_idr:
        end = len(disorder_values)
        if end - start + 1 >= min_length:
            idrs.append((start, end))

    return idrs


def is_position_in_idr(position: int, idrs: List[Tuple[int, int]]) -> bool:
    """
    Check if a position is within any IDR.
    :param position: Position to check (1-indexed)
    :param idrs: List of (start, end) tuples for IDRs (1-indexed)
    :return: True if position is in any IDR, False otherwise
    """
    for start, end in idrs:
        if start <= position <= end:
            return True
    return False


def is_disordered_mutation(seq: str, idx: int) -> bool:
    """
    Check if a residue at a given index in a sequence is disordered.
    This function uses the MetaPredict library to predict disorder scores for the sequence
    :param seq: The protein sequence
    :param idx: The index of the residue to check
    :return: True if the residue is in a disordered region, False otherwise
    """
    disorder_values = meta.predict_disorder(seq)
    idrs = identify_idrs(disorder_values, threshold=DISORDERED_THRESHOLD, min_length=3)
    is_disordered = is_position_in_idr(idx + 1, idrs)
    return is_disordered


def add_is_disordered_mutation_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column indicating whether each mutation is in a disordered region.
    :param df: DataFrame with columns 'protein_sequence', 'mutant'.
    :return: DataFrame with added 'is_disordered_mutation' column
    """
    if 'is_disordered_mutation' in df.columns:
        return df

    df = df.copy()
    disordered_lst = []
    for _, row in df.iterrows():
        protein_seq = row['protein_sequence']
        mutation = row['mutant']
        aa_mut = process_mutation_name(mutation, offset=OFFSET)
        is_disordered = is_disordered_mutation(protein_seq, aa_mut.mut_idx)
        disordered_lst.append(is_disordered)
    df['is_disordered_mutation'] = disordered_lst
    return df


def find_mask_positions(protein_seq: str, mask_token: str) -> List[Tuple[int, int]]:
    """
    Find all positions of mask tokens in the protein sequence.
    :param protein_seq: The protein sequence
    :param mask_token: The mask token string (e.g., "<mask>")
    :return: List of (start, end) tuples for mask positions
    """
    mask_positions = []
    start_pos = 0

    while True:
        mask_pos = protein_seq.find(mask_token, start_pos)
        if mask_pos == -1:
            break
        mask_positions.append((mask_pos, mask_pos + len(mask_token)))
        start_pos = mask_pos + len(mask_token)

    return mask_positions


def would_split_mask(chunk_start, chunk_end, mask_positions):
    """
    Check if chunk boundaries would split any mask token.

    Args:
        chunk_start: Start position of the chunk
        chunk_end: End position of the chunk
        mask_positions: List of (start, end) tuples for mask positions

    Returns:
        Tuple of (would_split, mask_start, mask_end)
    """
    for mask_start, mask_end in mask_positions:
        # Check if mask is partially inside the chunk (would be split)
        if (chunk_start < mask_end and chunk_end > mask_start and
                not (chunk_start <= mask_start and chunk_end >= mask_end)):
            return True, mask_start, mask_end
    return False, None, None


def calculate_chunk_positions(seq_length, chunk_size, overlap_size, mask_positions):
    """
    Calculate optimal chunk positions that don't split mask tokens.

    Args:
        seq_length: Total length of the sequence
        chunk_size: Maximum size of each chunk
        overlap_size: Size of overlap between chunks
        mask_positions: List of mask token positions

    Returns:
        List of dictionaries with chunk position information
    """
    chunk_positions = []
    start_pos = 0

    while start_pos < seq_length:
        end_pos = min(start_pos + chunk_size, seq_length)

        # Check if this chunk would split a mask
        would_split, mask_start, mask_end = would_split_mask(start_pos, end_pos, mask_positions)

        if would_split:
            # Adjust chunk_end to not split the mask
            if mask_start >= start_pos:
                # Mask starts within or after chunk start - end chunk before mask
                end_pos = mask_start
            else:
                # Mask starts before chunk start but ends within chunk
                # Include the entire mask
                end_pos = mask_end

        # If the adjusted chunk is too small, include the mask
        if end_pos - start_pos < overlap_size and would_split:
            for mask_start, mask_end in mask_positions:
                if mask_start < start_pos + chunk_size and mask_end > end_pos:
                    end_pos = min(mask_end, seq_length)
                    break

        # Skip empty chunks
        if end_pos <= start_pos:
            start_pos += 1
            continue

        # Define the valid region (excluding padding/overlap)
        valid_start = overlap_size if start_pos > 0 else 0
        valid_end = (end_pos - start_pos) - overlap_size if end_pos < seq_length else (end_pos - start_pos)

        # Ensure valid_end doesn't exceed chunk size and we have a valid region
        valid_end = min(valid_end, end_pos - start_pos)
        if valid_end <= valid_start:
            valid_end = end_pos - start_pos

        chunk_positions.append({
            'chunk_start': start_pos,
            'chunk_end': end_pos,
            'valid_start': valid_start,
            'valid_end': valid_end
        })

        # Break if we've reached the end
        if end_pos >= seq_length:
            break

        # Calculate next start position with overlap
        next_start = start_pos + chunk_size - 2 * overlap_size

        # Ensure we don't start in the middle of a mask token
        for mask_start, mask_end in mask_positions:
            if mask_start < next_start < mask_end:
                next_start = mask_end
                break

        start_pos = next_start

    return chunk_positions


def process_chunks_and_combine_logits(protein_seq, chunk_positions, alphabet, seq_name, model):
    """
    Process each chunk through the model and combine the logits.

    Args:
        protein_seq: The full protein sequence
        chunk_positions: List of chunk position dictionaries
        alphabet: The ESM alphabet
        seq_name: Name of the sequence
        model: The ESM model

    Returns:
        Combined logits tensor for the entire sequence
    """
    final_logits = None

    for i, pos in enumerate(chunk_positions):
        # Extract the chunk
        chunk_seq = protein_seq[pos['chunk_start']:pos['chunk_end']]

        # Process the chunk through the model
        batch_tokens = get_batch_token(alphabet, seq_name, chunk_seq)
        chunk_logits = get_trunctad_logits(True, batch_tokens, model)
        chunk_logits = chunk_logits.squeeze(0)

        # Extract the valid region (excluding overlaps)
        valid_logits = chunk_logits[pos['valid_start']:pos['valid_end']]

        # Combine with previous logits
        if final_logits is None:
            final_logits = valid_logits
        else:
            final_logits = torch.cat([final_logits, valid_logits], dim=0)

    return final_logits


def process_long_sequence_chunking_with_overlapping_regions(alphabet, seq_name, protein_seq, model):
    """
    Process long protein sequences by chunking with overlapping regions.
    Ensures that mask tokens are never split across chunks.

    Args:
        alphabet: The ESM alphabet
        seq_name: Name of the sequence
        protein_seq: The full protein sequence
        model: The ESM model

    Returns:
        Combined logits for the entire sequence
    """
    seq_length = len(protein_seq)

    mask_positions = find_mask_positions(protein_seq, MASK_TOKEN)

    chunk_positions = calculate_chunk_positions(
        seq_length,
        ESM_MAX_LENGTH,
        OVERLAP_SIZE_LONG_PROTEIN,
        mask_positions
    )

    final_logits = process_chunks_and_combine_logits(
        protein_seq,
        chunk_positions,
        alphabet,
        seq_name,
        model
    )

    return final_logits


def get_batch_token(alphabet, example_name, sequence):
    tokenizer = alphabet.get_batch_converter()
    input = [(example_name, sequence)]
    _, _, batch_tokens = tokenizer(input)
    batch_tokens = batch_tokens.to(DEVICE)
    return batch_tokens


def get_trunctad_logits(aa_only, batch_tokens, model):
    chunk_logits = model(batch_tokens, repr_layers=REP_LAYERS, return_contacts=False)['logits']
    logit_parts = []
    if ESM1B_MODEL == 'esm1_t6_43M_UR50S':
        logit_parts.append(chunk_logits[0, 1:, 4:24] if aa_only else chunk_logits[0, 1:, :])
    else:
        logit_parts.append(chunk_logits[0, 1:-1, 4:24] if aa_only else chunk_logits[0, 1:-1, :])
    return torch.stack(logit_parts).to(DEVICE)


def run_esm(model, alphabet, protein_seq: str, aa_mut: AAMut):
    model.eval()
    with torch.no_grad():
        seq_name = f"{aa_mut.wt_aa}{aa_mut.mut_idx}{aa_mut.mut_idx}"
        truncated_logits = process_long_sequence_chunking_with_overlapping_regions(alphabet, seq_name, protein_seq,
                                                                                   model)

        mutation_record = MutationRecord(
            protein_seq=protein_seq,
            aa_mut=aa_mut,
            truncated_logits=truncated_logits,
        )

    return mutation_record.mutant_log_marginal_probability.item(), mutation_record.marginal_entropy.item()


def _create_simple_bins(scores, n_bins):
    """Create bins by dividing data into equal-sized bins."""
    n_samples = len(scores)
    samples_per_bin = n_samples // n_bins

    sorted_scores = np.sort(scores)
    bin_edges = [sorted_scores[0] - 1e-10]

    for i in range(samples_per_bin, n_samples, samples_per_bin):
        if i < n_samples and len(bin_edges) < n_bins:
            boundary = (sorted_scores[i - 1] + sorted_scores[i]) / 2
            bin_edges.append(boundary)

    bin_edges.append(sorted_scores[-1] + 1e-10)
    return np.array(bin_edges), len(bin_edges) - 1


def create_unified_calibration(
        real_data: Optional[pd.DataFrame] = None,
        synthetic_data: Optional[Dict] = None,
        gmm_models_by_score: Optional[Dict] = None,
        score_col: str = None,
        n_bins: int = 20,
        samples_per_bin: Optional[int] = None,
        n_synthetic_samples: int = 30000,
        # TreeSpec parameters
        tree_spec = None,
        # Data combination strategy
        combination_strategy: str = 'auto'  # 'real_only', 'synthetic_only', 'combined', 'auto'
) -> Dict:
    """
    Unified calibration function that can handle:
    1. Real data only calibration
    2. GMM synthetic data only calibration
    3. Combined real + synthetic data calibration
    4. Auto-detection based on available data

    Args:
        real_data: DataFrame with real training data (columns: score_col, 'binary_label')
        synthetic_data: Dict with 'scores' and 'labels' arrays for synthetic data
        gmm_models_by_score: Dict of GMM models organized by score column
        score_col: Score column name to calibrate
        n_bins: Number of bins to create (if samples_per_bin not specified)
        samples_per_bin: Samples per bin (if specified, overrides n_bins)
        n_synthetic_samples: Number of synthetic samples to generate from GMM
        tree_spec: TreeSpec object defining the tree structure
        combination_strategy: How to combine data ('real_only', 'synthetic_only', 'combined', 'auto')

    Returns:
        Dict: Calibration information that can be used with apply_unified_calibration
    """
    print(f"Using combination strategy: {combination_strategy}")

    combined_scores = synthetic_data['scores']
    combined_labels = synthetic_data['labels']

    # Determine number of bins
    if samples_per_bin is not None:
        actual_n_bins = len(combined_scores) // samples_per_bin
        actual_n_bins = max(1, actual_n_bins)  # At least 1 bin
    else:
        actual_n_bins = min(n_bins, len(combined_scores) // 5)  # At least 5 samples per bin

    # Create bin edges
    bin_edges, actual_bins = _create_simple_bins(combined_scores, actual_n_bins)

    # Calculate pathogenic percentage per bin
    bin_stats = {}
    for idx in range(len(bin_edges) - 1):
        in_bin = (combined_scores >= bin_edges[idx]) & (combined_scores < bin_edges[idx + 1])
        if idx == len(bin_edges) - 2:  # Last bin includes right edge
            in_bin = (combined_scores >= bin_edges[idx]) & (combined_scores <= bin_edges[idx + 1])

        bin_labels = combined_labels[in_bin]
        if len(bin_labels) > 0:
            pathogenic_pct = (bin_labels.sum() / len(bin_labels)) * 100
            bin_stats[idx] = pathogenic_pct
        else:
            bin_stats[idx] = 50.0  # Default if no data in bin

    return {
        'bin_edges': bin_edges,
        'bin_stats': bin_stats,
        'actual_bins': actual_bins,
        'score_col': score_col,
        'combination_strategy': combination_strategy,
        'total_samples': len(combined_scores),
        'n_bins_created': actual_bins
    }


def apply_unified_calibration(
        test_data: pd.DataFrame,
        score_col: str,
        calibration: Dict,
        default_score: float = 50.0
) -> np.ndarray:
    """
    Apply unified calibration to test data.

    Args:
        test_data: DataFrame with test data
        score_col: Score column name to calibrate
        calibration: Calibration dict from create_unified_calibration
        default_score: Default score for missing bins (default: 50.0)

    Returns:
        Array of calibrated scores (0-100 scale)
    """
    if calibration is None:
        return np.full(len(test_data), default_score)

    scores = test_data[score_col].values
    calibrated_scores = np.full(len(test_data), np.nan)

    # Handle missing scores
    valid_mask = ~np.isnan(scores)
    if not valid_mask.any():
        raise ValueError("No valid scores to calibrate")

    # Assign scores to bins
    bin_indices = np.digitize(scores[valid_mask], calibration['bin_edges']) - 1
    bin_indices = np.clip(bin_indices, 0, len(calibration['bin_edges']) - 2)

    # Apply calibration
    valid_calibrated = np.full(valid_mask.sum(), default_score)
    for i, bin_idx in enumerate(bin_indices):
        if bin_idx in calibration['bin_stats']:
            valid_calibrated[i] = calibration['bin_stats'][bin_idx]

    calibrated_scores[valid_mask] = valid_calibrated
    # calibrated_scores[~valid_mask] = default_score

    return calibrated_scores


def calculate_best_threshold_metrics(labels, scores, confusion_matrix_metric_type: str, max_thresholds=100):
    """
    Calculate confusion matrix metrics at the best threshold based on specified metric.
    Automatically handles score orientation if needed.

    Returns tp, tn, fp, fn, accuracy, f1_score, j_score at best threshold.

    Args:
        labels: True binary labels (0=benign, 1=pathogenic)
        scores: Prediction scores
        confusion_matrix_metric_type: Either 'f1_score' or 'j_score' to determine which metric to optimize
        max_thresholds: Maximum number of thresholds to evaluate
        auto_orient: If True, automatically detect and correct score orientation
    """
    from sklearn.metrics import roc_curve, f1_score, accuracy_score, confusion_matrix, roc_auc_score
    import numpy as np

    # Validate metric_type
    if confusion_matrix_metric_type not in ['f1_score', 'j_score']:
        raise ValueError("metric_type must be either 'f1_score' or 'j_score'")
    # additional_metric_type = 'f1_score' if confusion_matrix_metric_type != 'f1_score' else 'j_score'

    # Handle score orientation
    original_scores = scores.copy()
    orientation_reversed = False

    initial_auc = roc_auc_score(labels, scores)
    if initial_auc < 0.5:
        # Reverse scores if AUC < 0.5
        scores = -scores
        orientation_reversed = True

    # Get thresholds from ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # If we have too many thresholds, sample them
    if len(thresholds) > max_thresholds:
        # Keep first, last, and evenly sample the middle
        indices = np.linspace(0, len(thresholds) - 1, max_thresholds, dtype=int)
        thresholds = thresholds[indices]

    best_score = -np.inf  # Allow negative J scores
    best_metrics = None

    best_f1_score = -np.inf
    best_j_score = -np.inf
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()


        f1 = f1_score(labels, predictions, zero_division=0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        j_score = sensitivity + specificity - 1
        # Choose which metric to optimize
        if confusion_matrix_metric_type == 'f1_score':
            current_score = f1
        else:  # metric_type == 'j_score'
            current_score = j_score

        original_threshold = -threshold if orientation_reversed else threshold

        if f1 > best_f1_score:
            best_f1_score = f1
        if j_score > best_j_score:
            best_j_score = j_score

        if current_score > best_score:
            best_score = current_score
            accuracy = accuracy_score(labels, predictions)

            total = tp + tn + fp + fn
            best_metrics = {
                'tp': float(tp / total * 100) if total > 0 else 0.0,
                'tn': float(tn / total * 100) if total > 0 else 0.0,
                'fp': float(fp / total * 100) if total > 0 else 0.0,
                'fn': float(fn / total * 100) if total > 0 else 0.0,
                'accuracy': float(accuracy),
                'j_score': float(j_score),
                'f1_score': float(f1),
                'best_threshold': float(original_threshold),
                'best_threshold_on_working_scale': float(threshold),
                'optimization_metric': confusion_matrix_metric_type,
                'orientation_reversed': orientation_reversed,
                'final_auc': float(roc_auc_score(labels, scores))
            }

    return best_metrics


def calculate_auc_and_thresholds(test_data, label_col, score_col, calibrated_col_name):
    labels = test_data[label_col].values
    calibrated_scores = test_data[calibrated_col_name].values
    if np.isnan(calibrated_scores).any():
        raise ValueError("Calibrated scores contain NaN values, cannot compute AUC")
    fpr_calibrated, tpr_calibrated, _ = roc_curve(labels, calibrated_scores)
    calibrated_auc = auc(fpr_calibrated, tpr_calibrated)
    print(f"Calibrated AUC for {calibrated_col_name}: {calibrated_auc:.3f}")

    raw_score = test_data[score_col].values
    fpr_raw, tpr_raw, _ = roc_curve(labels, raw_score)
    raw_auc = auc(fpr_raw, tpr_raw)
    raw_auc = raw_auc if raw_auc >= 0.5 else 1 - raw_auc
    print(f"Raw AUC for {score_col}: {raw_auc:.3f}")

    raw_score = test_data[score_col].values
    metric_type = 'j_score'
    threshold_cm = calculate_best_threshold_metrics(labels, calibrated_scores, confusion_matrix_metric_type=metric_type, max_thresholds=100)
    raw_cm = calculate_best_threshold_metrics(labels, raw_score, confusion_matrix_metric_type=metric_type, max_thresholds=100)
    print(threshold_cm)
    print(raw_cm)


def add_raw_esm1b_score(df: pd.DataFrame, protein_sequence_modified: str, score_col: str, entropy_score_col: str) -> pd.DataFrame:
    """
    Adding raw ESM1b scores and entropy scores to the DataFrame.
    :param df: Input DataFrame with columns 'protein_sequence' and 'mutant'
    :param protein_sequence_modified: Name of the column containing protein sequences
    :param score_col: Name of the column to store ESM1b scores
    :param entropy_score_col: Name of the column to store ESM1b entropy scores
    :return: DataFrame with added score columns
    """
    df = df.copy()
    if score_col in df.columns and entropy_score_col in df.columns:
        return df

    model, alphabet = esm_setup(ESM1B_MODEL, device=DEVICE)
    esm1b_scores = []
    esm1b_entropies = []
    for idx, row in df.iterrows():
        protein_seq = row[protein_sequence_modified]
        mutation = row['mutant']
        aa_mut = process_mutation_name(mutation, offset=OFFSET)
        esm1b_score, esm1b_entropy = run_esm(model, alphabet, protein_seq, aa_mut)
        esm1b_scores.append(esm1b_score)
        esm1b_entropies.append(esm1b_entropy)

    df[score_col] = esm1b_scores
    df[entropy_score_col] = esm1b_entropies
    return df


def add_is_ppi_mutation_column(df: pd.DataFrame, zip_path: str, json_filename_inside_zip: str) -> pd.DataFrame:
    """
    Add a column indicating whether each mutation affects a protein-protein interaction (PPI).
    Works with multiple protein sequences in the same DataFrame.
    :param df: Input DataFrame with columns 'protein_sequence', 'mutant', etc.
    :param zip_path: Path to the zip file containing PPI predictions in JSON format
    :param json_filename_inside_zip: Filename of the JSON file inside the zip
    :return: DataFrame with added 'is_ppi_mutation' column
    """
    if 'is_ppi_mutation' in df.columns:
        return df

    df_with_ppi = df.copy()

    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(json_filename_inside_zip) as f:
            ppi_predictions = json.load(f)

    df_with_ppi['is_ppi_mutation'] = np.nan

    sequences_found = 0
    sequences_not_found = 0
    mutations_mapped = 0
    mutations_unmapped = 0

    for idx, row in df_with_ppi.iterrows():
        sequence = row['protein_sequence']
        mutant = row['mutant']
        aa_mut = process_mutation_name(mutant, OFFSET)
        mutant_index = aa_mut.mut_idx

        # Check if sequence exists in PPI predictions
        if sequence in ppi_predictions:
            sequences_found += 1
            predictions_list = ppi_predictions[sequence]

            if 0 <= mutant_index < len(predictions_list):
                prediction_value = predictions_list[mutant_index]

                # Convert prediction to boolean
                # Values 1-3 are considered PPI, 0 is not PPI
                if prediction_value in [1, 2, 3]:
                    df_with_ppi.loc[idx, 'is_ppi_mutation'] = True
                    mutations_mapped += 1
                elif prediction_value == 0:
                    df_with_ppi.loc[idx, 'is_ppi_mutation'] = False
                    mutations_mapped += 1
                else:
                    # Unexpected value - leave as NaN
                    mutations_unmapped += 1
                    print(
                        f"Warning: Unexpected prediction value {prediction_value} for sequence {sequence[:20]}... at index {mutant_index}")
            else:
                # Index out of bounds - leave as NaN
                mutations_unmapped += 1
        else:
            # Sequence not found in predictions - leave as NaN
            sequences_not_found += 1
            mutations_unmapped += 1

    print(f"\nPPI Mapping Summary:")
    print(f"  Sequences found in PPI predictions: {sequences_found}")
    print(f"  Sequences not found: {sequences_not_found}")
    print(f"  Mutations successfully mapped: {mutations_mapped}")
    print(f"  Mutations not mapped: {mutations_unmapped}")

    return df_with_ppi


def clean_duplication_in_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove entries with conflicting binary labels for the same (protein_sequence, mutant) key.
    :param df:  Input DataFrame with columns 'protein_sequence', 'mutant', 'binary_label'
    :return: Cleaned DataFrame with duplicates removed
    """
    key = ['protein_sequence', 'mutant']

    # Step 1: find keys where there are conflicting binary labels
    diff_label_keys = (
        df
        .groupby(key)['binary_label']
        .nunique()
        .reset_index()
        .query('binary_label > 1')  # more than 1 unique label
    )

    # Step 2: remove those keys from df
    clean_df = (
        df
        .merge(diff_label_keys[key], on=key, how='left', indicator=True)
        .query('_merge == "left_only"')
        .drop(columns=['_merge'])
    )

    # Step 3: remove exact duplicates, keeping first occurrence
    clean_df = clean_df.drop_duplicates(subset=['protein_sequence', 'mutant'])

    return clean_df
