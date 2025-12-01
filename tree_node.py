import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from consts import MIN_VARIANTS_PER_LEAF, MIN_PATHOGENIC_PER_LEAF, MIN_BENIGN_PER_LEAF, BENIGN, PATHOGENIC
from data_classes import Dimension


Key = Tuple[str, ...]


@dataclass
class TreeNode:
    """
    A node in the calibration tree.
    """
    node_id: int
    parent: Optional['TreeNode'] = None
    children: Dict[str, 'TreeNode'] = field(default_factory=dict)
    split_dimension: Optional[Dimension] = None
    split_jsd: Optional[float] = None
    available_dimensions: List[Dimension] = field(default_factory=list)
    data_mask: Optional[pd.Series] = None
    leaf_key: Optional[Key] = None
    is_leaf: bool = True
    depth: int = 0
    has_calibration: bool = False
    calibration_key: Optional[Key] = None

    def meets_minimum_requirements(self, df: pd.DataFrame, label_col: str,
                                   min_variants_per_leaf: int, min_pathogenic_per_leaf: int, min_benign_per_leaf: int
                                   ) -> bool:
        """
        Check if this node meets minimum data requirements.
        :param df: The full DataFrame
        :param label_col: The name of the label column
        :param min_variants_per_leaf: A minimum total variants required
        :param min_pathogenic_per_leaf: A minimum pathogenic variants required
        :param min_benign_per_leaf: A minimum benign variants required
        :return: True if requirements are met, False otherwise
        """
        if self.data_mask is None:
            return False

        node_data = df[self.data_mask]
        n_total = len(node_data)
        n_pathogenic = node_data[label_col].sum()
        n_benign = n_total - n_pathogenic

        return (n_total >= min_variants_per_leaf and
                n_pathogenic >= min_pathogenic_per_leaf and
                n_benign >= min_benign_per_leaf)


class CalibrationTree:
    """
    The calibration tree structure.
    A dynamic tree that partitions data based on jensen-shannon divergence of class-conditional distributions.
    This tree is the calibration specification used to build calibration models.
    """
    available_dims: List[Dimension]
    length_dim: Optional[Dimension]
    path_prefix: str = "all"
    key_prefix: str = "ld"
    uniref_version: str = "uniref90"
    title_info: str = ""
    min_variants_per_leaf: int = MIN_VARIANTS_PER_LEAF
    min_pathogenic_per_leaf: int = MIN_PATHOGENIC_PER_LEAF
    min_benign_per_leaf: int = MIN_BENIGN_PER_LEAF

    # Tree structure (populated after building)
    root: Optional[TreeNode] = None
    leaf_nodes: List[TreeNode] = field(default_factory=list)
    calibration_nodes: List[TreeNode] = field(default_factory=list)
    dims: List[Dimension] = field(default_factory=list)

    _final_calibration_cache: Optional[Dict[int, Key]] = None

    def __init__(self, available_dims: List[Dimension], title_info, length_dim: Optional[Dimension] = None):
        self.available_dims = available_dims
        self.length_dim = length_dim
        self.title_info = title_info

    def build_tree(self, df: pd.DataFrame, score_col: str, label_col: str) -> 'CalibrationTree':
        # Initialize root with length split
        self.root = TreeNode(
            node_id=0,
            data_mask=pd.Series(True, index=df.index),
            available_dimensions=self.available_dims.copy(),
            depth=0
        )

        used_dims = []

        # Split root by length if length_dim is provided
        if self.length_dim is not None:
            print(f"\nRoot split: {self.length_dim.name}")
            self._split_node(self.root, self.length_dim, df)
            used_dims.append(self.length_dim)
        else:
            print("\nNo length split - starting with available dimensions")

        # Recursively build tree
        node_counter = [1]  # Use list to allow modification in nested function
        self._build_recursive(self.root, df, score_col, label_col, used_dims, node_counter)

        # Prune tree (Algorithm 2)
        self._prune_tree(df, label_col)
        self._update_leaf_and_calibration_status(df, label_col)

        # Collect final leaf nodes and build dims list
        self._finalize_tree(used_dims)

        self._compute_final_calibration_assignments(df)

        print("Tree Construction Complete")
        print(f"Total leaves: {len(self.leaf_nodes)}")
        print(f"Dimensions used: {[d.name for d in self.dims]}")
        print(f"Tree depth: {max(leaf.depth for leaf in self.leaf_nodes)}")

        return self

    def _update_leaf_and_calibration_status(self, df: pd.DataFrame, label_col: str):
        """
        Update is_leaf and has_calibration status after pruning.
        Ensures both leaf nodes AND parent nodes get calibration when appropriate.
        """

        def _update_node_recursive(node: TreeNode):
            # First, process all children
            for child in node.children.values():
                _update_node_recursive(child)

            # Update is_leaf status based on whether node has children
            node.is_leaf = len(node.children) == 0

            # Give calibration to nodes that meet requirements
            if node.meets_minimum_requirements(df, label_col, self.min_variants_per_leaf,
                                               self.min_pathogenic_per_leaf, self.min_benign_per_leaf):
                node.has_calibration = True
                node.calibration_key = self._build_leaf_key(node)
            else:
                node.has_calibration = False
                node.calibration_key = None

            # SPECIAL CASE: For single-child scenarios
            # Give calibration to intermediate nodes so they can serve as fallback
            if len(node.children) == 1:
                # The parent should have calibration to serve as fallback for missing branches
                if node.meets_minimum_requirements(df, label_col, self.min_variants_per_leaf,
                                                   self.min_pathogenic_per_leaf, self.min_benign_per_leaf):
                    node.has_calibration = True
                    node.calibration_key = self._build_leaf_key(node)

        if self.root is not None:
            _update_node_recursive(self.root)

    def _build_recursive(self, node: TreeNode, df: pd.DataFrame, score_col: str,
                         label_col: str, used_dims: List[Dimension], node_counter: List[int]) -> None:
        """
        Recursive function to build the tree. The tree is built by splitting nodes based on the highest JSD of the
        remaining available dimensions.
        :param node: Current TreeNode to process. Should be the root initially.
        :param df: The full DataFrame
        :param score_col: The name of the score column
        :param label_col: The name of the label column
        :param used_dims: List of dimensions that have been used so far
        :param node_counter: A list with a single integer to track node IDs
        :return: None. It updates the tree in place.
        """
        # Process children if they exist
        if not node.is_leaf:
            for child_name, child_node in node.children.items():
                self._build_recursive(child_node, df, score_col, label_col, used_dims, node_counter)
            return

        # Check if we can still split
        if not node.available_dimensions:
            return

        # Find best dimension to split on
        best_dim, best_jsd = self._find_best_split(node, df, score_col, label_col)

        if best_dim is None:
            return

        print(f"\nDepth {node.depth + 1}: Splitting on '{best_dim.name}' (JSD: {best_jsd:.4f})")

        # Split the node
        self._split_node(node, best_dim, df)
        node.split_jsd = best_jsd

        # Track dimension usage
        if best_dim not in used_dims:
            used_dims.append(best_dim)

        # Assign node IDs to children
        for child in node.children.values():
            child.node_id = node_counter[0]
            node_counter[0] += 1

        # Check each child individually
        for child_name, child_node in node.children.items():
            child_meets_req = child_node.meets_minimum_requirements(
                df, label_col, self.min_variants_per_leaf,
                self.min_pathogenic_per_leaf, self.min_benign_per_leaf
            )

            if child_meets_req:
                # Child can be split further, recurse
                self._build_recursive(child_node, df, score_col, label_col, used_dims, node_counter)
            else:
                # Child doesn't meet requirements - mark it but don't recurse
                child_node.is_leaf = True
                child_node.has_calibration = False

    def _find_best_split(self, node: TreeNode, df: pd.DataFrame,
                         score_col: str, label_col: str) -> Tuple[Optional[Dimension], float]:
        """Find the dimension that maximizes class-conditional distribution difference."""

        node_data = df[node.data_mask]

        if len(node_data) < self.min_variants_per_leaf:
            return None, 0.0

        best_dim = None
        best_jsd = -1

        for dim in node.available_dimensions:
            # Calculate JSD for this dimension
            jsd = self._calculate_jsd(node_data, dim, score_col, label_col)

            if jsd > best_jsd:
                best_jsd = jsd
                best_dim = dim

        return best_dim, best_jsd

    def _calculate_jsd(self, data: pd.DataFrame, dim: Dimension,
                       score_col: str, label_col: str, n_bins: int = 100) -> float:
        """
        Calculate Jensen-Shannon Divergence for class-conditional distributions.

        JSD = JSD(benign_positive, benign_negative) + JSD(pathogenic_positive, pathogenic_negative)
        """

        # Get positive and negative selections
        positive_mask = dim.values[dim.order[1]](data)  # e.g., "disordered", "sulfur", "ppi"

        # Split by class
        benign_data = data[data[label_col] == BENIGN]
        pathogenic_data = data[data[label_col] == PATHOGENIC]

        # Get scores for each subgroup
        benign_pos = benign_data[positive_mask[benign_data.index]][score_col].dropna().values
        benign_neg = benign_data[~positive_mask[benign_data.index]][score_col].dropna().values
        pathogenic_pos = pathogenic_data[positive_mask[pathogenic_data.index]][score_col].dropna().values
        pathogenic_neg = pathogenic_data[~positive_mask[pathogenic_data.index]][score_col].dropna().values

        # Check if we have enough data
        if len(benign_pos) < 10 or len(benign_neg) < 10 or \
                len(pathogenic_pos) < 10 or len(pathogenic_neg) < 10:
            return 0.0

        # Calculate common range for all histograms
        all_scores = np.concatenate([benign_pos, benign_neg, pathogenic_pos, pathogenic_neg])
        score_range = (all_scores.min(), all_scores.max())

        # Calculate JSD for each class using the helper function
        _, jsd_benign = jsd_from_samples_hist(benign_pos, benign_neg, bins=n_bins, range=score_range)
        _, jsd_pathogenic = jsd_from_samples_hist(pathogenic_pos, pathogenic_neg, bins=n_bins, range=score_range)

        # Sum JSDs
        total_jsd = jsd_benign + jsd_pathogenic

        return total_jsd

    def _split_node(self, node: TreeNode, dim: Dimension, df: pd.DataFrame):
        """Split a node on the given dimension."""

        node.is_leaf = False
        node.split_dimension = dim

        # Create child nodes for each value
        for value in dim.order:
            value_mask = dim.values[value](df[node.data_mask])
            child_mask = node.data_mask.copy()
            child_mask[node.data_mask] = value_mask

            # Remove used dimension from available dimensions
            remaining_dims = [d for d in node.available_dimensions if d.name != dim.name]

            child = TreeNode(
                node_id=-1,  # Will be assigned later
                parent=node,
                data_mask=child_mask,
                available_dimensions=remaining_dims,
                depth=node.depth + 1,
                is_leaf=True
            )

            node.children[value] = child

    def _prune_tree(self, df: pd.DataFrame, label_col: str):
        """
        Algorithm 2: Prune leaves that don't meet minimum requirements.
        Handle single-child scenarios properly - allow nodes to have only one child.
        """

        def _prune_recursive(node: TreeNode) -> bool:
            """
            Recursively prune tree. Returns True if node should be kept.
            """
            if node.is_leaf:
                # Leaf nodes are kept if they meet requirements
                meets_req = node.meets_minimum_requirements(df, label_col,
                                                            self.min_variants_per_leaf,
                                                            self.min_pathogenic_per_leaf,
                                                            self.min_benign_per_leaf)
                return meets_req

            # For non-leaf nodes, prune children first
            children_to_remove = []
            for child_key, child_node in node.children.items():
                if not _prune_recursive(child_node):
                    children_to_remove.append(child_key)

            # Remove pruned children
            for child_key in children_to_remove:
                del node.children[child_key]

            # If node has no children left, it becomes a leaf
            if len(node.children) == 0:
                node.is_leaf = True
                node.split_dimension = None
                node.split_jsd = None
                # Check if this new leaf meets requirements
                return node.meets_minimum_requirements(df, label_col,
                                                       self.min_variants_per_leaf,
                                                       self.min_pathogenic_per_leaf,
                                                       self.min_benign_per_leaf)

            # If node has exactly one child, we can either:
            # 1. Keep the split (your preference based on the question)
            # 2. Or collapse it
            # Based on your requirement, we'll keep nodes with single children

            # Node has children, so keep it
            return True

        if self.root is not None:
            print("Starting tree pruning...")
            initial_leaves = len(self._get_all_leaves(self.root))

            _prune_recursive(self.root)

            final_leaves = len(self._get_all_leaves(self.root))
            print(f"Pruning complete: {initial_leaves} -> {final_leaves} leaves")

    def _get_all_leaves(self, node: TreeNode) -> List[TreeNode]:
        """Recursively collect all leaf nodes."""

        if node.is_leaf:
            return [node]

        leaves = []
        for child in node.children.values():
            leaves.extend(self._get_all_leaves(child))

        return leaves

    def _finalize_tree(self, used_dims: List[Dimension]):
        """Collect final leaf nodes and calibration nodes."""

        self.dims = used_dims
        self.leaf_nodes = self._get_all_leaves(self.root)
        self.calibration_nodes = self._get_all_calibration_nodes(self.root)

        # Assign leaf keys
        for leaf in self.leaf_nodes:
            leaf.leaf_key = self._build_leaf_key(leaf)

        # Assign calibration keys
        for node in self.calibration_nodes:
            if node.calibration_key is None:
                node.calibration_key = self._build_leaf_key(node)

        print("\nFinal tree structure:")
        for i, leaf in enumerate(self.leaf_nodes):
            print(f"  Leaf {i}: {' -> '.join([str(v) for v in leaf.leaf_key])}")

        print(f"\nTotal calibration nodes: {len(self.calibration_nodes)}")
        print(f"Total leaf nodes: {len(self.leaf_nodes)}")

    def _calculate_cache_path(self, df: pd.DataFrame):
        base_path = "/cs/labs/dina/sapir_amittai/code/lab_winter_25/save_datasets_tmp/final_calibration_cache_cache"
        dims_as_str = "_".join([d.name for d in self.dims])
        return os.path.join(
            base_path,
            f"cache_{len(df)}_min_var_{self.min_variants_per_leaf}_min_path_{self.min_pathogenic_per_leaf}_dims_{dims_as_str}.pkl"
        )

    def _compute_final_calibration_assignments(self, df: pd.DataFrame):
        """
        Compute and cache which calibration node each row in df should use.
        This is called once during build_tree and cached for fast lookup.
        Cache uses protein_sequence + mutant as unique key.
        """
        cache_final_calibration_cache_path = self._calculate_cache_path(df)
        if os.path.exists(cache_final_calibration_cache_path):
            with open(cache_final_calibration_cache_path, 'rb') as file:
                self._final_calibration_cache = pickle.load(file)
            return

        print("\nComputing final calibration assignments...")

        assigned_keys = {}  # maps "protein_mutant" -> calibration_key

        def assign_recursive(node: TreeNode, row_mask: pd.Series):
            """Recursively assign rows to their deepest calibration node."""
            if not row_mask.any():
                return

            # If this node has calibration, tentatively assign these rows to it
            if node.has_calibration:
                for idx in df.index[row_mask]:
                    # Create unique key from protein_sequence and mutant
                    protein = df.loc[idx, 'protein_sequence']
                    mutant = df.loc[idx, 'mutant']
                    unique_key = f"{protein}_{mutant}"
                    assigned_keys[unique_key] = node.calibration_key

            # If this is a leaf or has no split, we're done
            if node.is_leaf or node.split_dimension is None:
                return

            # Split rows based on dimension
            dim = node.split_dimension
            dim_data = df.loc[row_mask]

            for value_name, child_node in node.children.items():
                try:
                    # Apply dimension filter
                    child_filter = dim.values[value_name](dim_data)

                    # Create full-size mask
                    child_mask = pd.Series(False, index=df.index)
                    child_mask.loc[dim_data.index] = child_filter.values

                    # Continue to child if it has calibration or children
                    if child_node.has_calibration or not child_node.is_leaf:
                        assign_recursive(child_node, child_mask)

                except Exception:
                    # If evaluation fails, rows stay with parent calibration
                    continue

        # Start from root
        root_mask = pd.Series(True, index=df.index)
        assign_recursive(self.root, root_mask)

        # Store cache
        self._final_calibration_cache = assigned_keys

        with open(cache_final_calibration_cache_path, 'wb') as f:
            pickle.dump(self._final_calibration_cache, f)
        print(f"Saved calibration cache to {cache_final_calibration_cache_path}")
        print(f"Cached calibration assignments for {len(assigned_keys)} unique mutations")

    def _get_all_calibration_nodes(self, node: TreeNode) -> List[TreeNode]:
        """Recursively collect all nodes that have calibration data."""

        calibration_nodes = []

        # Add this node if it has calibration
        if node.has_calibration:
            calibration_nodes.append(node)

        # Recurse on children
        if not node.is_leaf:
            for child in node.children.values():
                calibration_nodes.extend(self._get_all_calibration_nodes(child))

        return calibration_nodes

    def _build_leaf_key(self, leaf: TreeNode) -> Key:
        """Build the key for a leaf by traversing up to root."""

        path = []
        current = leaf

        while current.parent is not None:
            # Find which value led to this node
            for value, child in current.parent.children.items():
                if child is current:
                    path.append(value)
                    break
            current = current.parent

        return tuple(reversed(path))


def dim_length(threshold: int):
    return Dimension(
        name="length",
        col_name="sequence_length",
        values={
            "short": lambda df: df["sequence_length"] <= threshold,
            "long": lambda df: df["sequence_length"] > threshold,
        },
        order=["short", "long"],
        label_map={"short": f"shorter_than_{threshold}", "long": f"longer_than_{threshold}"},
    )


def dim_fold():
    return Dimension(
        name="fold",
        col_name="is_disordered_mutation",
        values={
            "ordered": lambda df: df["is_disordered_mutation"] == False,
            "disordered": lambda df: df["is_disordered_mutation"] == True,
        },
        order=["ordered", "disordered"],
        label_map={"ordered": "ordered", "disordered": "disordered"},
    )


def dim_ppi():
    return Dimension(
        name="ppi",
        col_name="is_ppi_mutation",
        values={
            "non_ppi": lambda df: df["is_ppi_mutation"] == False,
            "ppi": lambda df: df["is_ppi_mutation"] == True,
        },
        order=["non_ppi", "ppi"],
        label_map={"non_ppi": "non_ppi", "ppi": "ppi"},
    )


def dim_sulfur():
    return Dimension(
        name="sulfur",
        col_name="mutant",
        values={
            "non_sulfur": lambda df: ~df["mutant"].str.match(r'^[CM]', na=False),
            "sulfur": lambda df: df["mutant"].str.match(r'^[CM]', na=False),
        },
        order=["non_sulfur", "sulfur"],
        label_map={"non_sulfur": "non_sulfur", "sulfur": "sulfur"},
    )


def dim_homologs(threshold: int):
    return Dimension(
        name="homologs",
        col_name="number_of_same_sequence_in_cluster",
        values={
            "low": lambda df: df["number_of_same_sequence_in_cluster"] <= threshold,
            "high": lambda df: df["number_of_same_sequence_in_cluster"] > threshold,
        },
        order=["low", "high"],
        label_map={
            "low": f"homologs_0_to_{threshold}",
            "high": f"homologs_{threshold}_plus",
        },
    )


def build_node_calibration_masks(df: pd.DataFrame, spec: CalibrationTree) -> Dict[Key, pd.Series]:
    """
    Build masks for calibration nodes where each mask includes all data in the subtree.

    Works with any DataFrame (subset or different data) by computing masks dynamically,
    similar to how build_leaf_masks works.

    Only includes calibration nodes that are ACTUALLY USED for calibration.

    Returns:
        Dict mapping calibration_key -> mask of all data in that subtree
    """
    if spec.root is None:
        return {}

    # First, get the actual usage from build_leaf_masks for this specific df
    leaf_masks = build_leaf_masks(df, spec)
    # return leaf_masks
    actually_used_calibration_keys = set()

    # Find which calibration keys are actually used (have non-zero masks)
    for calibration_key, mask in leaf_masks.items():
        if mask.sum() > 0:
            actually_used_calibration_keys.add(calibration_key)

    # Initialize masks only for actually used calibration nodes
    calibration_masks = {}

    def _build_node_mask_for_df(node: TreeNode, current_mask: pd.Series) -> pd.Series:
        """
        Build mask for this node by filtering the current_mask through the tree path.
        Returns mask of data from input df that belongs to this node.
        """
        if not current_mask.any():
            return pd.Series(False, index=df.index)

        # If this is a leaf or has no split, return current mask
        if node.is_leaf or node.split_dimension is None:
            return current_mask.copy()

        # For non-leaf nodes, combine masks from all children PLUS unmatched data
        combined_mask = pd.Series(False, index=df.index)
        dim = node.split_dimension
        dim_data = df.loc[current_mask]

        # Track which rows get assigned to children
        assigned_to_children = pd.Series(False, index=df.index)

        # Process each child
        for value_name, child_node in node.children.items():
            try:
                # Apply dimension filter
                child_filter = dim.values[value_name](dim_data)

                # Create child mask
                child_mask = pd.Series(False, index=df.index)
                child_mask.loc[dim_data.index] = child_filter.values

                # Track assigned rows
                assigned_to_children |= child_mask

                # Get child's subtree mask
                child_subtree_mask = _build_node_mask_for_df(child_node, child_mask)
                combined_mask |= child_subtree_mask

            except Exception:
                continue

        # Add unmatched data (missing branches) to this node's mask
        unmatched_mask = current_mask & ~assigned_to_children
        combined_mask |= unmatched_mask

        return combined_mask

    def _collect_calibration_masks_recursive(node: TreeNode, current_mask: pd.Series):
        """
        Recursively collect calibration masks for nodes that are actually used.
        """
        if not current_mask.any():
            return

        # Only include this node if it's actually used for calibration
        if (node.has_calibration and
                node.calibration_key is not None and
                node.calibration_key in actually_used_calibration_keys):
            node_mask = _build_node_mask_for_df(node, current_mask)
            calibration_masks[node.calibration_key] = node_mask

        # Continue to children if not a leaf
        if not node.is_leaf and node.split_dimension is not None:
            dim = node.split_dimension
            dim_data = df.loc[current_mask]

            for value_name, child_node in node.children.items():
                try:
                    # Apply dimension filter
                    child_filter = dim.values[value_name](dim_data)

                    # Create child mask
                    child_mask = pd.Series(False, index=df.index)
                    child_mask.loc[dim_data.index] = child_filter.values

                    # Continue to child
                    _collect_calibration_masks_recursive(child_node, child_mask)

                except Exception:
                    continue

    # Start collection from root with all data
    root_mask = pd.Series(True, index=df.index)
    _collect_calibration_masks_recursive(spec.root, root_mask)

    return calibration_masks


def build_leaf_masks(df: pd.DataFrame, spec: CalibrationTree) -> Dict[Key, pd.Series]:
    """
        Build masks showing the FINAL calibration node for each mutation.
        ULTRA-FAST VERSION: Uses pre-computed cache with pandas merge (fully vectorized).

        Uses protein_sequence + mutant as unique key for cache lookup.
        """
    if spec.root is None:
        return {}

    # Check if required columns exist
    if 'protein_sequence' not in df.columns or 'mutant' not in df.columns:
        print("Warning: 'protein_sequence' or 'mutant' columns not found. Computing from scratch.")
        return _compute_masks_from_scratch(df, spec)

    # Initialize masks
    masks = {node.calibration_key: pd.Series(False, index=df.index)
             for node in spec.calibration_nodes if node.calibration_key is not None}

    # Check if we can use cache
    if spec._final_calibration_cache is not None and len(spec._final_calibration_cache) > 0:
        # Convert cache to DataFrame for fast merge
        cache_data = []
        for unique_key, calibration_key in spec._final_calibration_cache.items():
            # Split back to protein and mutant
            parts = unique_key.rsplit('_', 1)  # rsplit in case protein has underscores
            if len(parts) == 2:
                protein, mutant = parts
                cache_data.append({
                    'protein_sequence': protein,
                    'mutant': mutant,
                    '_calibration_key': calibration_key
                })

        cache_df = pd.DataFrame(cache_data)

        # Add temporary index column to df for tracking
        df_temp = df.copy()
        df_temp['_original_index'] = df_temp.index

        # Merge to get calibration keys - FAST!
        merged = df_temp[['_original_index', 'protein_sequence', 'mutant']].merge(
            cache_df,
            on=['protein_sequence', 'mutant'],
            how='left'
        )

        # Count cache hits
        cache_hits = merged['_calibration_key'].notna().sum()
        total = len(df)
        print(f"Cache hits: {cache_hits}/{total} ({100 * cache_hits / total:.1f}%)")
        if 100 * cache_hits / total != 100:
            ...

        # Assign to masks using VECTORIZED operations - NO LOOP!
        for calibration_key in masks.keys():
            # Find all rows with this calibration key
            matching_rows = merged[merged['_calibration_key'] == calibration_key]
            if len(matching_rows) > 0:
                # Set all matching indices to True at once
                matching_indices = matching_rows['_original_index'].values
                masks[calibration_key].loc[matching_indices] = True

        # Handle cache misses if any
        cache_miss_rows = merged[merged['_calibration_key'].isna()]
        if len(cache_miss_rows) > 0:
            cache_misses = cache_miss_rows['_original_index'].values
            print(f"Computing calibration for {len(cache_misses)} new mutations...")
            miss_df = df.loc[cache_misses]
            new_assignments = _compute_assignments_for_df(miss_df, spec)

            for idx, key in new_assignments.items():
                if key in masks:
                    masks[key].loc[idx] = True

        return masks

    # No cache available - compute from scratch
    print(f"No cache available. Computing calibration for {len(df)} mutations...")
    return _compute_masks_from_scratch(df, spec)


def _compute_masks_from_scratch(df: pd.DataFrame, spec: CalibrationTree) -> Dict[Key, pd.Series]:
    """Compute masks from scratch without using cache."""
    masks = {node.calibration_key: pd.Series(False, index=df.index)
             for node in spec.calibration_nodes if node.calibration_key is not None}

    assignments = _compute_assignments_for_df(df, spec)

    for idx, key in assignments.items():
        if key in masks:
            masks[key].loc[idx] = True

    return masks


def _compute_assignments_for_df(df: pd.DataFrame, spec: CalibrationTree) -> Dict[int, Key]:
    """Helper function to compute calibration assignments for a DataFrame."""
    assigned_keys = {}

    def assign_recursive(node: TreeNode, row_mask: pd.Series):
        if not row_mask.any():
            return

        if node.has_calibration:
            for idx in df.index[row_mask]:
                assigned_keys[idx] = node.calibration_key

        if node.is_leaf or node.split_dimension is None:
            return

        dim = node.split_dimension
        dim_data = df.loc[row_mask]

        for value_name, child_node in node.children.items():
            try:
                child_filter = dim.values[value_name](dim_data)
                child_mask = pd.Series(False, index=df.index)
                child_mask.loc[dim_data.index] = child_filter.values

                if child_node.has_calibration or not child_node.is_leaf:
                    assign_recursive(child_node, child_mask)

            except Exception:
                continue

    root_mask = pd.Series(True, index=df.index)
    assign_recursive(spec.root, root_mask)

    return assigned_keys


def key_to_name(spec: CalibrationTree, key, sep="_") -> str:
    """Convert leaf key to readable name."""
    return sep.join(key)


def print_tree(node, prefix="", is_last=True, df=None):
    connector = "└── " if is_last else "├── "

    if node.is_leaf:
        if df is not None:
            node_data = df[node.data_mask]
            n_pathogenic = node_data[node_data['binary_label'] == PATHOGENIC].shape[0]
            n_benign = node_data[node_data['binary_label'] == BENIGN].shape[0]
            print(f"{prefix}{connector}LEAF: {node.leaf_key} [n={len(node_data)} n_path={n_pathogenic} n_ben={n_benign}]")
        else:
            print(f"{prefix}{connector}LEAF: {node.leaf_key}")
    else:
        split_name = node.split_dimension.name
        jsd_str = f"(JSD: {node.split_jsd:.4f})" if node.split_jsd is not None else ""  # ADD THIS LINE
        n_pathogenic = df[node.data_mask][df[node.data_mask]['binary_label'] == PATHOGENIC].shape[0] if df is not None else 0
        n_benign = df[node.data_mask][df[node.data_mask]['binary_label'] == BENIGN].shape[0] if df is not None else 0
        amount_on_split = f"[n={len(df[node.data_mask])} n_path={n_pathogenic} n_ben={n_benign}]"
        print(f"{prefix}{connector}Split on: {split_name} {jsd_str} {amount_on_split}")  # MODIFY THIS LINE

        children = list(node.children.items())
        for i, (value, child) in enumerate(children):
            is_last_child = (i == len(children) - 1)
            extension = "    " if is_last else "│   "
            print(f"{prefix}{extension}[{value}]")
            print_tree(child, prefix + extension, is_last_child, df)


def jsd_from_samples_hist(x, y, bins=100, range=None, base=2, eps=1e-12):
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
