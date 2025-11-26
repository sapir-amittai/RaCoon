from typing import Optional

import torch
from os.path import join as pjoin

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#  DIRECTORIES

HOMOLOGS_PATH = pjoin('data', 'homologs_dict.pickle')
CALIBRATION_HISTOGRAMS = pjoin('data', 'alibration_histograms')

#  VARIANT PROCESSING

VALID_AA = "ACDEFGHIKLMNPQRSTVWY"
STOP_AA = '_'
N_AA = len(VALID_AA)
AA_SYN = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE",
          "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER",
          "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"}
AA_SYN_REV = dict((v, k) for k, v in AA_SYN.items())
AA_TO_INDEX_ESM = {'K': 0, 'R': 1, 'H': 2, 'E': 3, 'D': 4, 'N': 5, 'Q': 6, 'T': 7, 'S': 8, 'C': 9, 'G': 10,
                   'A': 11, 'V': 12, 'L': 13, 'I': 14, 'M': 15, 'P': 16, 'Y': 17, 'F': 18, 'W': 19}
MUTATION_REGEX =  rf'p\.(?P<symbol>(?P<orig>[{VALID_AA}]){{1}}(?P<location>[\d]+)(?P<change>[{VALID_AA}]){{1}})'

ESM_AA_ORDER = 'LAGVSERTIDPKQNFYMHWC'
ESM_AA_LOC = {aa: idx for idx, aa in enumerate(ESM_AA_ORDER)}

#  VARIANT PREDICTION & ESM

ESM1B_MODEL = 'esm1b_t33_650M_UR50S'
REP_LAYERS = [33]
# ESM1B_MODEL = 'esm1_t6_43M_UR50S'
# REP_LAYERS = [6]
ESM_MAX_LENGTH = 1022
MASK_TOKEN = '<mask>'

DISORDERED_THRESHOLD = 0.7
OVERLAP_SIZE_LONG_PROTEIN = 250

PATHOGENIC = 1
BENIGN = 0

LONG = 'long'
SHORT = 'short'
DISORDERED = 'disordered'
ORDERED = 'ordered'
SULFUR = 'sulfur'
NON_SULFUR = 'non_sulfur'
PPI = 'ppi'
NON_PPI = 'non_ppi'

SCORE_COL = 'wt_not_nadav_marginals_base_wt_score'
WT_ENTROPY_COL_NAME = 'wt_record_entropy'
PPI_PREDICTIONS_PATH = "pioneer_predictions.zip"
PPI_PREDICTIONS_FILE_NAME = "pioneer_predictions.json"


MIN_VARIANTS_PER_LEAF = 1600
MIN_PATHOGENIC_PER_LEAF = 400
MIN_BENIGN_PER_LEAF = 400
N_BINS = 50
N_SAMPLES_TO_GENERATE = 40000
GMM_CREATION_SAMPLE_SIZE = 400
RANDOM_STATE = 42
RATIO_SAMPLE_SIZE = 500
OFFSET = 1
