"""
Pruning utilities for tree-based structures.

This module provides functions for identifying and pruning leaf nodes in tree structures,
useful for Graph SSM models that build and traverse tree structures.

OPTIMIZED VERSION: Fully vectorized operations using CUDA-accelerated tensor operations.
"""

import torch


def find_leaf_nodes_vectorized(tree):
    """
    Find leaf nodes in the tree structure using fully vectorized operations.
    
    Args:
        tree: Tensor [batch, num_edges, 2] where each edge is [source, target]
    
    Returns:
        leaf_nodes_mask: Boolean tensor [batch, num_nodes] where True = leaf node
        leaf_edges_mask: Boolean tensor [batch, num_edges] where True = edge connected to leaf
    """
    batch_size, num_edges, _ = tree.shape
    num_nodes = num_edges + 1  # Tree property: N nodes, N-1 edges
    device = tree.device
    
    # Convert to int64 for vectorized operations (tree comes in as int32 for CUDA kernels)
    tree_long = tree.long()
    
    # Vectorized degree calculation
    # Create edge indices for scatter operations
    src_nodes = tree_long[:, :, 0]  # [batch, num_edges]
    dst_nodes = tree_long[:, :, 1]  # [batch, num_edges]
    
    # Use scatter_add to count degrees efficiently
    # Initialize degree tensor
    degree = torch.zeros(batch_size, num_nodes, dtype=torch.int32, device=device)
    
    # Count degrees using scatter_add (CUDA-accelerated)
    # src_nodes and dst_nodes are already int64 from tree_long
    degree.scatter_add_(1, src_nodes, torch.ones_like(src_nodes, dtype=torch.int32))
    degree.scatter_add_(1, dst_nodes, torch.ones_like(dst_nodes, dtype=torch.int32))
    
    # Leaf nodes have degree 1
    leaf_nodes_mask = (degree == 1)  # [batch, num_nodes]
    
    # Find edges connected to leaf nodes (vectorized)
    # Check if either source or destination is a leaf
    # src_nodes and dst_nodes are already int64 from tree_long
    src_is_leaf = leaf_nodes_mask.gather(1, src_nodes)  # [batch, num_edges]
    dst_is_leaf = leaf_nodes_mask.gather(1, dst_nodes)   # [batch, num_edges]
    leaf_edges_mask = src_is_leaf | dst_is_leaf  # [batch, num_edges]
    
    return leaf_nodes_mask, leaf_edges_mask


def prune_leaf_nodes_vectorized(tree, edge_weights=None, num_leaves_to_prune=1, pruning_mode='ordered', verbose=False):
    """
    Create a mask for edges to prune based on leaf nodes using fully vectorized operations.
    
    Args:
        tree: Tensor [batch, num_edges, 2]
        edge_weights: Optional tensor [batch, num_edges] of edge weights from MST
        num_leaves_to_prune: Number of leaf nodes to remove
        pruning_mode: 'ordered' to prune most similar nodes (highest weights), 
                     'unordered' to prune randomly
        verbose: Whether to print debug information
    
    Returns:
        edge_mask: Boolean tensor [batch, num_edges] where False = pruned, True = keep
        num_removed: Number of edges marked for removal per batch
    """
    batch_size, num_edges, _ = tree.shape
    device = tree.device
    
    # Initialize edge mask (all edges kept by default)
    edge_mask = torch.ones(batch_size, num_edges, dtype=torch.bool, device=device)
    
    if num_leaves_to_prune <= 0:
        return edge_mask, torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    # Find leaf nodes and leaf edges using vectorized operations
    leaf_nodes_mask, leaf_edges_mask = find_leaf_nodes_vectorized(tree)
    
    # Count leaf edges per batch
    num_leaf_edges_per_batch = leaf_edges_mask.sum(dim=1)  # [batch]
    
    # Handle batches with no leaf edges
    valid_batches = num_leaf_edges_per_batch > 0
    
    if not valid_batches.any():
        return edge_mask, torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    # For batches with leaf edges, perform pruning
    if pruning_mode == 'unordered':
        # Random pruning: randomly select leaf edges to prune
        prune_mask = torch.ones_like(edge_mask)
        batch_indices = torch.arange(batch_size, device=device)
        
        for b in range(batch_size):
            if num_leaf_edges_per_batch[b] > 0:
                # Get indices of leaf edges for this batch
                leaf_edge_indices = torch.where(leaf_edges_mask[b])[0]
                num_available = len(leaf_edge_indices)
                num_to_prune = min(num_leaves_to_prune, num_available)
                
                if num_to_prune > 0:
                    # Randomly select edges to prune
                    perm = torch.randperm(num_available, device=device)
                    edges_to_prune = leaf_edge_indices[perm[:num_to_prune]]
                    prune_mask[b, edges_to_prune] = False
        
        # Apply pruning mask
        edge_mask = edge_mask & prune_mask
        
    else:  # 'ordered' mode - prune based on similarity (highest weights)
        if edge_weights is not None:
            # Use provided edge weights
            pruning_weights = edge_weights.clone()
        else:
            # Use uniform weights if not provided
            pruning_weights = torch.ones_like(leaf_edges_mask, dtype=torch.float32)
        
        # Mask weights to only consider leaf edges
        leaf_weights = pruning_weights * leaf_edges_mask.float()
        
        # Set non-leaf edge weights to -inf so they won't be selected
        leaf_weights = torch.where(leaf_edges_mask, leaf_weights, torch.tensor(float('-inf'), device=device))
        
        # Find top-k leaf edges to prune (vectorized across batches)
        # We need to handle variable number of leaf edges per batch
        max_leaf_edges = num_leaf_edges_per_batch.max().item()
        
        if max_leaf_edges > 0:
            # Get top-k leaf edges for pruning
            _, top_indices = torch.topk(leaf_weights, min(num_leaves_to_prune, max_leaf_edges), dim=1)
            
            # Create pruning mask for top-k edges
            prune_mask = torch.ones_like(edge_mask)
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
            
            # Only prune if we have enough leaf edges
            for k in range(min(num_leaves_to_prune, max_leaf_edges)):
                # Get the k-th highest weight leaf edge for each batch
                edge_indices = top_indices[:, k]  # [batch]
                
                # Only prune if this batch has enough leaf edges
                valid_for_pruning = (num_leaf_edges_per_batch > k) & valid_batches
                
                if valid_for_pruning.any():
                    # Set mask to False (prune) for selected edges
                    prune_mask[batch_indices[valid_for_pruning], edge_indices[valid_for_pruning]] = False
            
            # Apply pruning mask
            edge_mask = edge_mask & prune_mask
    
    # Calculate number of edges removed per batch
    num_removed_per_batch = (leaf_edges_mask & ~edge_mask).sum(dim=1)
    
    if verbose:
        # Print debug info (only for first batch to avoid spam)
        if batch_size > 0:
            print(f"   Batch 0: Found {num_leaf_edges_per_batch[0].item()} leaf edges")
            print(f"   Pruned {num_removed_per_batch[0].item()} edges ({pruning_mode} mode)")
    
    return edge_mask, num_removed_per_batch


# Backward compatibility - keep original functions but mark as deprecated
def find_leaf_nodes(tree):
    """
    DEPRECATED: Use find_leaf_nodes_vectorized instead for better performance.
    """
    leaf_nodes_mask, leaf_edges_mask = find_leaf_nodes_vectorized(tree)
    
    # Convert to old format for backward compatibility
    batch_size = tree.shape[0]
    leaf_nodes_batch = []
    leaf_edges_batch = []
    
    for b in range(batch_size):
        leaf_nodes = torch.where(leaf_nodes_mask[b])[0].cpu().tolist()
        leaf_edges = torch.where(leaf_edges_mask[b])[0].cpu().tolist()
        leaf_nodes_batch.append(leaf_nodes)
        leaf_edges_batch.append(leaf_edges)
    
    return leaf_nodes_batch, leaf_edges_batch


def prune_leaf_nodes(tree, edge_weights=None, num_leaves_to_prune=1, pruning_mode='ordered', verbose=False):
    """
    DEPRECATED: Use prune_leaf_nodes_vectorized instead for better performance.
    """
    edge_mask, num_removed_per_batch = prune_leaf_nodes_vectorized(
        tree, edge_weights, num_leaves_to_prune, pruning_mode, verbose
    )
    
    # Convert to old format for backward compatibility
    num_removed_list = num_removed_per_batch.cpu().tolist()
    
    return edge_mask, num_removed_list

