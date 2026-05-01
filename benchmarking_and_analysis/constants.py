# GLOBAL CONSTANTS
AA = 'ACDEFGHIKLMNPQRSTVWY'

ALL_RESIDUE_SUBGROUPS =  ['is_disordered', 'low_homologs', 'is_ppi', 'is_polar', 'is_hydrophobic', 'is_charged',
                          'is_aromatic','is_acidic','is_basic','is_polar_uncharged','is_small','is_sulfur',
                          'is_proline_or_glycine','is_aliphatic','is_helix_breaker','is_beta_branched']

# Figure specific constants
MODELS_FOR_1A = {
    'MutationTaster': 'supervised',
    'FATHMM': 'supervised',
    'DANN': 'supervised',
    'LIST-S2': 'unsupervised',
    'PrimateAI': 'unsupervised',
    'DEOGEN2': 'supervised',
    'SIFT': 'unsupervised',
    'CADD': 'supervised',
    'MPC': 'supervised',
    'PROVEAN': 'unsupervised',
    'Polyphen2_HDIV': 'supervised',
    'ESM1b': 'unsupervised',
    'Polyphen2_HVAR': 'supervised',
    'MutationAssessor': 'unsupervised',
    'BayesDel_noAF': 'supervised',
    'REVEL': 'supervised',
    'gMVP': 'supervised',
    'ESM1v': 'unsupervised',
    'VEST4': 'supervised',
    'EVE': 'unsupervised',
    'BayesDel_addAF': 'supervised',
    'ClinPred': 'supervised',
    'AlphaMissense': 'semi_supervised',
    'MetaRNN': 'supervised',
    'CPT-1':'Semi-supervised',
}

MODELS_FOR_1B = ['EVE', 'GEMME', 'MutationAssessor', 'PoET', 'PrimateAI', 'Provean', 'SIFT', 'ESM1b']

MODELS_FOR_3_SUPERVISED = ['ClinPred', 'MetaRNN', 'BayesDel (addAF)', 'VEST4', 'REVEL', 'BayesDel (noAF)', 'VARITY (R)',
                         'VARITY (ER)', 'CPT-1']
MODELS_FOR_3_UNSUPERVISED = ['PrimateAI', 'MutationAssessor', 'LRT', 'SIFT', 'EVE', 'GEMME', 'ESM1b', 'PoET', 'LIST-S2',
                           'Provean', 'MutPred', 'TranceptEVE_L', 'SIFT4G']
MODELS_FOR_3 = MODELS_FOR_3_SUPERVISED + MODELS_FOR_3_UNSUPERVISED
FLIP_SCORES = {'PrimateAI': 1.0, 'MutationAssessor': 1.0, 'EVE': -1.0, 'Provean': -1.0, 'ESM1b': 1.0,
                   'LIST-S2': 1.0, 'LRT': -1.0, 'SIFT': -1.0, 'REVEL': 1.0, 'VEST4': 1.0, 'MetaRNN': 1.0,
                   'ClinPred': 1.0, 'BayesDel (addAF)': 1.0, 'BayesDel (noAF)': 1.0, 'VARITY (R)': 1.0}
RESIDUE_SUBGROUPS_FOR_3 = ['is_disordered', 'is_ppi', 'is_disordered:is_ppi']
CUSTOM_SELECTION =  {'is_disordered': None, 'is_ppi': None,
                     'is_disordered:is_ppi': [([0], [1]), ([0], [0]), ([1], [0, 1])]}
