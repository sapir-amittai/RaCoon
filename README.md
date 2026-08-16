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


### 
**Running with `clinvar_balanced.parquet`**

For convenience, `clinvar_balanced.parquet` includes the relevant precomputed columns required by the pipeline, including the ESM1b-based score, entropy, and the residue-level annotation `is_disordered_mutation`.

This means that RaCoon can be run directly with `clinvar_balanced.parquet` and produce results without recomputing these features, making execution faster and more convenient for repeated runs.

This file is provided only as a convenience option. Users do not need to supply `clinvar_balanced.parquet` specifically.

Example:

```bash
python main.py --df_path clinvar_balanced.parquet --save_df_path results/clinvar_balanced_racoon.parquet
```

**What this produces:**
Since `wt_score`, `wt_record_entropy`, and `is_disordered_mutation` are already present in `clinvar_balanced.parquet`, RaCoon skips ESM1b scoring and disorder prediction entirely and goes straight to building the calibration tree and calibrating scores. Console output includes the constructed calibration tree (splits and leaf sizes), per-leaf mutation counts, and the calibrated vs. raw AUC (e.g. `Calibrated AUC: 0.916` vs. `Raw AUC: 0.911`). The saved output file contains the input columns plus:

| Column name | Description |
| --- | --- |
| `racoon_pathogenic_probability` | The calibrated pathogenicity probability/score (see Output Interpretation below). |
| `racoon_node` | The calibration-tree leaf (subgroup) the variant was assigned to, e.g. `('short', 'ordered', 'non_sulfur', 'non_ppi')`. |

Note that the output only contains the held-out test rows (variants not used for calibration/training), not every input row - with the full `clinvar_balanced.parquet` (40,474 rows), this run produced 32,444 output rows.

**Expected run time:** Because the heavy steps (ESM1b scoring, disorder prediction) are skipped, this demo is fast - the full run (tree construction, GMM fitting, calibration, and saving results for all ~40K variants) completed in well under a minute (~15 seconds) on a standard CPU-only desktop. If you instead run RaCoon on data where `wt_score`/`wt_record_entropy` or `is_disordered_mutation` are missing, RaCoon computes them itself (loading ESM1b and/or running disorder prediction per variant), which is significantly slower.

###
**System Requirements** 

- **Operating system:** Tested on Linux. Not tested on Windows or macOS, but no OS-specific dependencies are used, so it is expected to work on any standard Linux/macOS setup.
- **Python:** 3.11.2 (the version used in the tested environment).
- **Dependencies:** Standard, widely-used Python packages - `numpy`, `pandas`, `pyarrow`, `scipy`, `scikit-learn`, `torch`, `metapredict` (exact versions pinned in `requirements.txt`, matching the environment RaCoon was developed and tested with). No non-standard or proprietary dependencies.
- **Hardware:** No non-standard hardware required, and **no GPU is required**. RaCoon was originally run on a GPU for faster ESM1b scoring, but the code automatically falls back to CPU when none is available. A GPU will speed up scoring of large batches of variants, but is not needed to run RaCoon.
- **Install time:** Installation typically takes a few minutes. The ESM1b model weights are downloaded automatically on first use.

###
**Installation and Usage**

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


###
**License**

RaCoon is released under the [MIT License](LICENSE), a permissive open-source license (OSI-approved) that allows use, modification, and redistribution with attribution.

### **Reference**
If you use this code, please cite our [paper](https://www.biorxiv.org/content/10.1101/2025.11.24.690189v1).
