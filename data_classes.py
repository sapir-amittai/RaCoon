from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Callable
import pandas as pd

BoolFunc = Callable[[pd.DataFrame], pd.Series]


class METHODS_TO_ESM(Enum):
    """
    Enum for mutation names
    """
    MUTANTE = "mutant_marginals"
    MASKED = "masked_marginals"
    WT = "wt_marginals"


@dataclass
class AAMut:
    """
    Data class representing an amino acid mutation.
    """
    wt_aa: str
    mut_idx: int
    change_aa: str


@dataclass(frozen=True)
class Dimension:
    """Represents a binary partitioning attribute."""
    name: str
    col_name: str
    values: Dict[str, BoolFunc]
    order: List[str]
    label_map: Dict[str, str]
