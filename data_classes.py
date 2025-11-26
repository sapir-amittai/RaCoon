from dataclasses import dataclass
from enum import Enum

from mutation_record import MutationRecord


class METHODS_TO_ESM(Enum):
    """
    Enum for mutation names
    """
    MUTANTE = "mutant_marginals"
    MASKED = "masked_marginals"
    WT = "wt_marginals"

@dataclass
class AAMut:
    wt_aa: str
    mut_idx: int
    change_aa: str
