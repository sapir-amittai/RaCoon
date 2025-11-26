from functools import cached_property
from typing import Optional, List, TYPE_CHECKING
import numpy as np
import torch
import math

from consts import ESM_AA_LOC, ESM_AA_ORDER


class MutationRecord:

    def __init__(self, protein_seq: str, aa_mut, truncated_logits: torch.Tensor):
        self.protein_seq = protein_seq
        self.aa_mut = aa_mut
        self.truncated_logits = truncated_logits

    @cached_property
    def entropy_tensor(self):
        probs = torch.softmax(self.relevant_logits, dim=-1)
        entropy_seqs = torch.distributions.Categorical(probs=probs).entropy()
        return entropy_seqs

    @cached_property
    def relevant_logits(self):
        # return MutationRecord.static_relevant_logits(self.truncated_logits, self.aa_mut)
        return self.truncated_logits[self.aa_mut.mut_idx] - self.truncated_logits[self.aa_mut.mut_idx][ESM_AA_LOC[self.aa_mut.wt_aa]]

    @cached_property
    def llr_logits(self):
        return torch.log_softmax(self.relevant_logits, dim=-1)

    @cached_property
    def llr_base_score(self):
        """
        The log-likelihood ratio of the mutant in the position of the mutation
        """
        return self.llr_logits[ESM_AA_LOC[self.aa_mut.change_aa]]
