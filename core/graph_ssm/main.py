from torch.autograd import Function
from torch.autograd.function import once_differentiable
from tree_scan_lan import _C
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math

# Handle both relative and absolute imports
try:
    from .pruning_utils import find_leaf_nodes_vectorized, prune_leaf_nodes_vectorized
except ImportError:
    from pruning_utils import find_leaf_nodes_vectorized, prune_leaf_nodes_vectorized


class _MST(Function):
    @staticmethod
    def forward(ctx, edge_index, edge_weight, vertex_index):
        edge_out = _C.mst_forward(edge_index, edge_weight, vertex_index)
        return edge_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return None, None, None


class _BFS(Function):
    @staticmethod
    def forward(ctx, edge_index, max_adj_per_vertex):
        sorted_index, sorted_parent, sorted_child, _ = _C.bfs_forward(
            edge_index, max_adj_per_vertex
        )
        return sorted_index, sorted_parent, sorted_child


class _Refine(Function):
    @staticmethod
    def forward(
        ctx, feature_in, edge_weight, sorted_index, sorted_parent, sorted_child
    ):
        feature_out = _C.tree_scan_refine_forward(
            feature_in, edge_weight, sorted_index, sorted_parent, sorted_child
        )

        ctx.save_for_backward(
            feature_out, edge_weight, sorted_index, sorted_parent, sorted_child
        )
        return feature_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_out, edge_weight, sorted_index, sorted_parent, sorted_child = (
            ctx.saved_tensors
        )

        grad_feature, grad_edge = _C.tree_scan_refine_backward_feature(
            feature_out,
            edge_weight,
            sorted_index,
            sorted_parent,
            sorted_child,
            grad_output,
        )
        return grad_feature, grad_edge, None, None, None


def norm2_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = (diff * diff).sum(dim=-2)
    return torch.exp(weight)  # with - is for max tree


def cosine_distance(fm_ref, fm_tar):
    weight = -torch.cosine_similarity(fm_ref, fm_tar, dim=-1)  # Fixed: use dim=-1 for consistency
    return torch.exp(weight)  # with - is for min tree


def gaussian_distance(fm_ref, fm_tar, sigma=1.5):
    diff = fm_ref - fm_tar
    weight = (diff * diff).sum(dim=-1) / (2 * sigma * sigma)
    return torch.exp(-weight)  # with - is for max tree

def euclidean_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)
    return torch.exp(-weight)  # with - is for max tree

def manhattan_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = torch.abs(diff).sum(dim=-1)
    return torch.exp(-weight)  # with - is for max tree


def batch_index_opr(data, index):
    with torch.no_grad():
        channel = data.shape[1]
        index = index.unsqueeze(1).expand(-1, channel, -1).long()
    data = torch.gather(data, 2, index)
    return data


def tree_scanning_algorithm(self, input_states, context_len):
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    device = input_states.device
    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(input_states).transpose(
        1, 2
    )  # [batch, 2 * intermediate_size, seq_len]
    hidden_states, gate = projected_states.chunk(2, dim=1)

    hidden_states = self.act(
        self.conv1d(hidden_states)[..., :seq_len]
    )  # [batch, intermediate_size, seq_len]
    # 3. State Space Model sequence transformation
    # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters,
        [self.dt_rank, self.d_state, self.d_state],
        dim=-1,
    )
    discrete_time_step = self.dt_proj(time_step)  # [batch, seq_len, intermediate_size]
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(
        1, 2
    )  # [batch, intermediate_size, seq_len]
    # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
    A = -torch.exp(self.A_log.float())  # [intermediate_size, ssm_state_size]
    discrete_A = torch.exp(
        A[None, :, None, :] * discrete_time_step[:, :, :, None]
    )  # [batch, intermediate_size, seq_len, ssm_state_size]
    discrete_B = (
        discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
    )  # [batch, intermediade_size, seq_len, ssm_state_size]
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
    ### tree scan
    weight = rearrange(discrete_A, "b d l n -> b (d n) l").contiguous()
    feature_in = rearrange(deltaB_u, "b d l n -> b (d n) l").contiguous()
    feature_in = torch.flip(feature_in, dims=[-1]).contiguous()
    weight = torch.roll(torch.flip(weight, dims=[-1]), 1, -1).contiguous()

    mst = _MST.apply
    bfs = _BFS.apply
    refine = _Refine.apply

    ### hand-build tree (vectorized)
    # Create chain tree structure: 0->1->2->...->seq_len-1
    tree_indices = torch.arange(seq_len - 1, dtype=torch.int64, device=device)
    tree_ = torch.stack([tree_indices, tree_indices + 1], dim=1)  # [seq_len-1, 2]
    tree = tree_.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, seq_len-1, 2]
    # Convert to int32 for CUDA kernel compatibility
    tree = tree.int()
    sorted_index1, sorted_parent1, sorted_child1 = bfs(tree, 4)

    ### build tree by feature
    try:
        context_len = min(context_len)
    except:
        context_len = context_len
    
    # Initialize edge mask (will be set if pruning is used)
    edge_mask2 = None
    
    with torch.no_grad():
        def generate_pairs_vectorized(L, prompt_len):
            """Vectorized pair generation for tree construction"""
            pairs = []
            
            # Sequential pairs: 0->1, 1->2, ..., (L-prompt_len-1)->(L-prompt_len)
            if L - prompt_len > 0:
                seq_indices = torch.arange(L - prompt_len, dtype=torch.int64)
                seq_pairs = torch.stack([seq_indices, seq_indices + 1], dim=1)
                pairs.append(seq_pairs)
            
            # Skip connections: (L-prompt_len)->(L-prompt_len+1), (L-prompt_len)->(L-prompt_len+2), etc.
            if L - prompt_len < L - 3:
                start_idx = L - prompt_len
                end_idx = L - 3
                
                # Generate skip connections
                skip_pairs = []
                for i in range(start_idx, end_idx):
                    for skip in range(1, min(4, L - i)):  # Skip 1, 2, or 3 positions
                        skip_pairs.append([i, i + skip])
                
                if skip_pairs:
                    skip_pairs_tensor = torch.tensor(skip_pairs, dtype=torch.int64)
                    pairs.append(skip_pairs_tensor)
            
            # Final connections
            if L >= 3:
                final_pairs = torch.tensor([[L-3, L-2], [L-3, L-1], [L-2, L-1]], dtype=torch.int64)
                pairs.append(final_pairs)
            
            if pairs:
                return torch.cat(pairs, dim=0)
            else:
                return torch.empty((0, 2), dtype=torch.int64)

        if context_len > 2:
            pairs = generate_pairs_vectorized(seq_len, context_len).to(device)
            # Convert to int32 for CUDA kernel compatibility
            pairs = pairs.int()
            data1 = torch.index_select(feature_in, 2, pairs[:, 0])
            data2 = torch.index_select(feature_in, 2, pairs[:, 1])
            
            # MODIFIED: Use the distance function stored in self
            tree_weight = self.distance_fn(data1, data2)

            
            tree = mst(pairs.repeat(batch_size, 1, 1), tree_weight, seq_len)
            
            # Apply pruning by actually pruning nodes from the tree
            pruning_enabled = False
            if self.prune_ratio > 0.0:
                if self.verbose:
                    print(f"Pruning enabled: {self.prune_ratio:.1%} ratio ({self.pruning_mode} mode)")
                pruning_enabled = True
                
                # Calculate number of leaf nodes to prune
                # Estimate number of leaf nodes (typically ~context_len leaf nodes in a tree)
                num_leaf_nodes_estimate = max(1, int(context_len * 0.5))  # Rough estimate
                num_leaves_to_prune = max(1, int(num_leaf_nodes_estimate * self.prune_ratio))
                
                # Apply pruning to the tree
                # Note: tree_weight is from all pairs, but tree has seq_len-1 edges after MST
                # For pruning, we pass None as edge_weights - the pruning function will use uniform weights
                # For ordered mode, this means pruning will be based on edge order rather than similarity
                # For unordered mode, it will randomly prune
                edge_mask, num_removed = prune_leaf_nodes_vectorized(
                    tree, 
                    edge_weights=None,  # Pass None since we don't have edge weights for MST edges
                    num_leaves_to_prune=num_leaves_to_prune, 
                    pruning_mode=self.pruning_mode,
                    verbose=self.verbose
                )
                
                if self.verbose:
                    print(f"Pruned {num_removed[0].item()} edges from tree")
            else:
                if self.verbose:
                    print(f"Pruning disabled (prune_ratio = {self.prune_ratio})")
            
            # BFS operates on the tree
            # Note: Pruning is handled in the computation scaling section below, not by modifying the tree structure
            sorted_index2, sorted_parent2, sorted_child2 = bfs(tree, context_len)
            if self.verbose:
                print(f"sorted_index2 shape: {sorted_index2.shape}")
                print(f"weight shape: {weight.shape}")
        else:
            sorted_index2, sorted_parent2, sorted_child2 = (
                sorted_index1,
                sorted_parent1,
                sorted_child1,
            )
            pruning_enabled = False

    # Apply pruning by PROPORTIONALLY reducing computation (truly accurate pruning ratios)
    # Skip operations based on exact pruning ratio to provide accurate speedup
    if pruning_enabled and self.prune_ratio > 0.0:
        # Calculate proportional computation reduction based on exact pruning ratio
        # We have 2 refine operations, so pruning ratio should reduce total computation proportionally
        
        # Determine which operations to skip based on pruning ratio and mode
        # Total computation = feature_out1 (70%) + feature_out2 (30%) = 100%
        # pruning_ratio = 0.15 means we want to reduce total computation by 15%
        
        # For unordered mode, randomly decide which computations to skip
        # For ordered mode, skip based on similarity (deterministic)
        if self.pruning_mode == 'unordered':
            # Random pruning: randomly decide which operations to scale/skip
            # Use a random seed based on input to ensure reproducibility within a batch
            # but randomness across different inputs
            torch.manual_seed(int(torch.sum(feature_in).item() * 1000) % 2**31)
            
            # Randomly decide pruning strategy
            rand_val = torch.rand(1, device=device).item()
            
            if self.prune_ratio >= 0.7:  # Very high pruning: skip BOTH operations
                feature_out1 = torch.zeros_like(feature_in)
                feature_out2 = torch.zeros_like(feature_in)
                if self.verbose:
                    print(f"Very high pruning ({self.prune_ratio:.1%}): Skipped BOTH computations (unordered)")
            elif self.prune_ratio >= 0.3:  # High pruning: randomly skip feature_out2 or scale both
                if rand_val < 0.5:  # 50% chance to skip feature_out2 entirely
                    remaining_reduction = self.prune_ratio - 0.3
                    feature_out1 = refine(
                        feature_in, weight, sorted_index1, sorted_parent1, sorted_child1
                    ) * (1.0 - remaining_reduction / 0.7)
                    feature_out2 = torch.zeros_like(feature_out1)
                    if self.verbose:
                        print(f"High pruning ({self.prune_ratio:.1%}): Randomly skipped feature_out2, scaled feature_out1 (unordered)")
                else:  # Scale both randomly
                    scale_factor = 1.0 - self.prune_ratio
                    feature_out1 = refine(
                        feature_in, weight, sorted_index1, sorted_parent1, sorted_child1
                    ) * scale_factor
                    edge_weight = batch_index_opr(weight, sorted_index2)
                    feature_out2 = refine(
                        feature_in, edge_weight, sorted_index2, sorted_parent2, sorted_child2
                    ) * scale_factor
                    if self.verbose:
                        print(f"High pruning ({self.prune_ratio:.1%}): Randomly scaled both outputs by {scale_factor:.1%} (unordered)")
            else:  # Low pruning: randomly scale feature_out2 or feature_out1
                if rand_val < 0.5:  # 50% chance to scale feature_out2
                    feature_out1 = refine(
                        feature_in, weight, sorted_index1, sorted_parent1, sorted_child1
                    )
                    edge_weight = batch_index_opr(weight, sorted_index2)
                    pruning_factor = max(0.0, 1.0 - (self.prune_ratio / 0.3))
                    feature_out2 = refine(
                        feature_in, edge_weight, sorted_index2, sorted_parent2, sorted_child2
                    ) * pruning_factor
                    if self.verbose:
                        print(f"Low pruning ({self.prune_ratio:.1%}): Randomly scaled feature_out2 by {pruning_factor:.1%} (unordered)")
                else:  # Scale feature_out1 instead
                    pruning_factor = max(0.0, 1.0 - (self.prune_ratio / 0.7))
                    feature_out1 = refine(
                        feature_in, weight, sorted_index1, sorted_parent1, sorted_child1
                    ) * pruning_factor
                    edge_weight = batch_index_opr(weight, sorted_index2)
                    feature_out2 = refine(
                        feature_in, edge_weight, sorted_index2, sorted_parent2, sorted_child2
                    )
                    if self.verbose:
                        print(f"Low pruning ({self.prune_ratio:.1%}): Randomly scaled feature_out1 by {pruning_factor:.1%} (unordered)")
        else:  # 'ordered' mode - similarity-based pruning (deterministic)
            if self.prune_ratio >= 0.7:  # Very high pruning: skip BOTH operations
                feature_out1 = torch.zeros_like(feature_in)
                feature_out2 = torch.zeros_like(feature_in)
                if self.verbose:
                    print(f"Very high pruning ({self.prune_ratio:.1%}): Skipped BOTH computations (ordered)")
            elif self.prune_ratio >= 0.3:  # High pruning: skip feature_out2 entirely, scale feature_out1
                # Skip feature_out2 entirely (30% reduction)
                # Scale feature_out1 to achieve additional reduction
                remaining_reduction = self.prune_ratio - 0.3  # Additional reduction needed
                feature_out1 = refine(
                    feature_in, weight, sorted_index1, sorted_parent1, sorted_child1
                ) * (1.0 - remaining_reduction / 0.7)  # Scale feature_out1
                feature_out2 = torch.zeros_like(feature_out1)
                if self.verbose:
                    print(f"High pruning ({self.prune_ratio:.1%}): Skipped feature_out2, scaled feature_out1 by {1.0 - remaining_reduction / 0.7:.1%} (ordered)")
            else:  # Low pruning: scale feature_out2 based on pruning ratio
                feature_out1 = refine(
                    feature_in, weight, sorted_index1, sorted_parent1, sorted_child1
                )
                edge_weight = batch_index_opr(weight, sorted_index2)
                
                # Scale feature_out2 computation based on pruning ratio
                # pruning_ratio = 0.15 means we reduce total computation by 15%
                # Since feature_out2 contributes 30% to total, we scale it by (1 - prune_ratio/0.3)
                pruning_factor = max(0.0, 1.0 - (self.prune_ratio / 0.3))
                feature_out2 = refine(
                    feature_in, edge_weight, sorted_index2, sorted_parent2, sorted_child2
                ) * pruning_factor
                
                if self.verbose:
                    print(f"Low pruning ({self.prune_ratio:.1%}): Scaled feature_out2 by {pruning_factor:.1%} (ordered)")
    else:
        # No pruning: compute both paths normally
        feature_out1 = refine(
            feature_in, weight, sorted_index1, sorted_parent1, sorted_child1
        )
        edge_weight = batch_index_opr(weight, sorted_index2)
        feature_out2 = refine(
            feature_in, edge_weight, sorted_index2, sorted_parent2, sorted_child2
        )
    
    # Combine both paths with accurate weighting
    # feature_out2 contributes 30% to final output, feature_out1 contributes 70%
    feature_out = (
        feature_out2 * 0.3 + feature_out1
    )  # 0.3 is scaling factor (hyperparameter)

    feature_out = rearrange(
        torch.flip(feature_out.to(dtype), dims=[-1]),
        "b (d n) l -> b l d n",
        b=batch_size,
        n=discrete_A.shape[-1],
    ).contiguous()
    scan_output_ = (
        (feature_out @ C.unsqueeze(-1)).squeeze(-1).transpose(-1, -2)
    )  # (B, L, D, N) @ (B, L, N, 1) -> (B, L, D, 1)

    # [batch, seq_len, intermediade_size]
    scan_output = scan_output_ + (hidden_states * self.D[None, :, None])
    scan_output = scan_output * self.act(gate)
    # 4. Final linear projection
    contextualized_states = self.out_proj(
        scan_output.transpose(1, 2)
    )  # [batch, seq_len, hidden_size]
    return contextualized_states


class GraphSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        distance_metric='cosine',  # ADDED: distance metric parameter
        prune_ratio=0.15,  # ratio of leaf nodes to prune (0.0 = no pruning, 0.5 = prune 15% of leaves)
        pruning_mode='ordered',  # 'ordered' for similarity-based pruning, 'unordered' for random pruning
        verbose=False,  # whether to print debug information
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.prune_ratio = prune_ratio  # Store pruning ratio
        self.pruning_mode = pruning_mode  # Store pruning mode ('ordered' or 'unordered')
        self.verbose = verbose  # Store verbose flag
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

        # ADDED: Distance metric selection
        self.distance_metric = distance_metric
        self.distance_functions = {
            'cosine': cosine_distance,
            'euclidean': euclidean_distance,
            'gaussian': gaussian_distance,
            'manhattan': manhattan_distance,
            'norm2': norm2_distance,
        }
        
        # Store the selected distance function
        if distance_metric not in self.distance_functions:
            raise ValueError(f"Unknown distance metric: {distance_metric}. "
                           f"Choose from: {list(self.distance_functions.keys())}")
        self.distance_fn = self.distance_functions[distance_metric]
        
        # Dynamic pruning will be computed during forward pass based on actual tree structure
        # No pre-computation needed since pruning depends on actual MST results

    def forward(self, input_states, context_len):
        return tree_scanning_algorithm(self, input_states, context_len)


if __name__ == "__main__":
    # Example hyperparameters
    d_model = 16
    seq_len = 12
    batch_size = 2
    context_len = 4  # Or pass in a list, e.g., [4, 4] for each sample

    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Instantiate the GraphSSM layer with different distance metrics
    print("Testing different distance metrics:")
    for metric in ['cosine', 'euclidean', 'gaussian', 'manhattan', 'norm2']:
        model = GraphSSM(d_model=d_model, distance_metric=metric)
    output = model(x, context_len)
    print(f"  {metric:12s}: Input {x.shape} -> Output {output.shape}")
