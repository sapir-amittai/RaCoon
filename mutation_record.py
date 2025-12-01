from functools import cached_property
import numpy as np
import torch

from consts import ESM_AA_LOC
from data_classes import AAMut


class MutationRecord:

    def __init__(self, protein_seq: str, aa_mut: AAMut, truncated_logits: torch.Tensor) -> None:
        """
        :param protein_seq: The full protein sequence.
        :param aa_mut: An AAMut object describing the mutation (index, wt aa, mutant aa).
        :param truncated_logits: A tensor of shape [..., 20] containing model logits for amino acids.
        """
        self.protein_seq = protein_seq
        self.aa_mut = aa_mut
        self.truncated_logits = truncated_logits

    @cached_property
    def position_logits_centered_on_wt(self):
        """
        Extract and center the logits at the mutation position by subtracting the WT amino acid logit.
        :return:
            A tensor of shape [20] where each entry corresponds to the centered logit
            for one amino acid at the mutation site.
        """
        return self.truncated_logits[self.aa_mut.mut_idx] - self.truncated_logits[self.aa_mut.mut_idx][ESM_AA_LOC[self.aa_mut.wt_aa]]

    @cached_property
    def marginal_entropy(self):
        """
        Compute the entropy of the amino-acid marginal distribution at the mutation site.
        This entropy quantifies the uncertainty of the model's distribution for the
        position, whether WT, mutant, or masked model inputs were used.
        :return:
            A scalar tensor representing the entropy of the marginal distribution
            at the mutation position. Shape: [] (0-dim tensor).
        """
        probs = torch.softmax(self.position_logits_centered_on_wt, dim=-1)
        entropy_seqs = torch.distributions.Categorical(probs=probs).entropy()
        return entropy_seqs

    @cached_property
    def log_marginals_probability(self):
        """
        Compute log-probabilities (log-marginals) over amino acids at the mutation position.
        This applies log_softmax to the centered logits, producing a normalized log-probability
        distribution over all 20 amino acids.
        :return:
            A tensor of shape [20] containing log-probabilities for each amino acid at the mutation site.
        """
        return torch.log_softmax(self.position_logits_centered_on_wt, dim=-1)

    @cached_property
    def mutant_log_marginal_probability(self):
        """
        Retrieve the log-marginal probability of the mutant amino acid at the mutation site.
        This value corresponds to log P(mutant_aa | sequence_context).
        :return:
            A scalar tensor representing the log-probability of the mutant amino acid.
            Shape: [] (0-dim tensor).
        """
        return self.log_marginals_probability[ESM_AA_LOC[self.aa_mut.change_aa]]
