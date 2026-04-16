**RaCoon (Residue-aware Calibration via Conditional distributions)**

###
**Overview**

RaCoon provides a calibrated, interpretable pathogenicity scoring framework for missense variants based on ESM1b. It adjusts variant effect predictions using residue-specific context to improve both probability calibration and ranking performance across diverse variant subgroups.

Pipeline Overview

1. Assign residue attributes:
Variants are annotated with residue-level properties (e.g., disorder, PPI involvement, sulfur-binding) and with technical context such as protein length.


2. Build a calibration tree (partitioning strategy):
Variants are first split by protein length (short vs. long). Then, the tree is expanded by iteratively splitting nodes using the attribute that yields the largest class-conditional distribution shift (measured by Jensen–Shannon divergence, JSD) between partitions.


3. Prune low-support nodes:
To ensure reliable subgroup estimates, the tree is pruned so that only leaves with sufficient labeled data are kept:
  MIN_VARIANTS_PER_LEAF = 1600, MIN_PATHOGENIC_PER_LEAF = 400, and MIN_BENIGN_PER_LEAF = 400.


4. Model pathogenic/benign score distributions:
For each retained leaf (subgroup), ESM1b scores* are modeled using Gaussian Mixture Models to estimate class-conditional score distributions.


5. Build calibration histograms:
Synthetic samples drawn from the GMMs are converted into histograms mapping raw scores to subgroup-specific pathogenicity estimates.


6. Calibrate variant scores:
Each variant is mapped to its leaf node and assigned a calibrated, interpretable pathogenicity probability based on its histogram bin.

*ESM1b scores correspond to the log-likelihood ratio (LLR) comparing the mutant amino acid to the wild-type amino acid at the variant position, derived from ESM1b model logits.

###
**Input / Output Overview**

RaCoon operates on a table provided via df_path and optionally saves the results to save_df_path.
Required columns in the df that in df_path are:

| Column name        | Example           | Description                                                                      |
| ------------------ | ----------------- |----------------------------------------------------------------------------------|
| `protein_sequence` | `"MEEPQSDPSV..."` | The full amino-acid sequence for the wild-type protein.       |
| `mutant`           | `"R175H"`         | Missense mutation encoded as `<wtAA><position_with_1_offset><mutatedAA>`.        |
| `binary_label`     | `1` or `0`        | Pathogenicity label: **1 = pathogenic**, **0 = benign**. Needed for GMM training. |

**Note about binary_label:**
Conceptually, RaCoon only requires labels for the calibration/training step.
However, this implementation expects the column to exist even when predicting.
For unlabeled variants, you may set the value to NaN or a dummy placeholder and exclude them during calibration.

Optional output:
1. save_df_path: if provided, the DataFrame augmented with RaCoon outputs will be written to this path.


### Running with `clinvar_balanced.parquet`

For convenience, `clinvar_balanced.parquet` includes the relevant precomputed columns required by the pipeline, including the ESM1b-based score, entropy, and the residue-level annotation `is_disordered_mutation`.

This means that RaCoon can be run directly with `clinvar_balanced.parquet` and produce results without recomputing these features, making execution faster and more convenient for repeated runs.

This file is provided only as a convenience option. Users do not need to supply `clinvar_balanced.parquet` specifically.

Example:

```bash
python main.py --df_path clinvar_balanced.parquet --save_df_path results/clinvar_balanced_racoon.parquet
```

###
**Python API (example)**

Create a virtual environment, install dependencies, and run RaCoon:

  ```
  python3 -m venv env
  source env/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  
  python main.py --df_path data/variants.csv --save_df_path (optional) results/variants_racoon.csv
  ```

###
**Output Interpretation**

The core result is a calibrated pathogenicity probability (racoon_pathogenic_probability) between 0 and 1.

Interpretation:
- A value of 0.8 indicates that, within similar residue contexts, roughly 80% of variants are expected to be pathogenic.
- Calibration is done per residue subgroup, not globally, ensuring more reliable probabilities across:
  - ordered vs. disordered regions
  - interface vs. non-interface residues
  - sulfur-binding residues vs others
  - short vs. long proteins


### **Reference**
If you use this code, please cite our [paper](https://www.biorxiv.org/content/10.1101/2025.11.24.690189v1).
