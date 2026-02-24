# Mesh-Based Hierarchical Profiling Configs

This directory contains example mesh-based hierarchical topology configurations for common cluster architectures.

## Overview

**Device mesh abstraction** enables intuitive network layer specification using logical dimensions (node, GPU, rack, etc.) instead of manual rank enumeration.

**Benefits:**
- ✅ **Clear semantics** - "vary GPU, fix node" = intra-node profiling
- ✅ **Auto-generation** - Ranks derived from mesh, no manual enumeration
- ✅ **Portable** - Same config works across cluster sizes (just change `shape`)
- ✅ **Validated** - System checks that rank pairs match mesh structure

## Config Files

### 1. `perlmutter_mesh.json`
**Topology:** NERSC Perlmutter
- **Mesh:** 4 nodes × 4 GPUs/node = 16 ranks
- **Dimensions:** `["node", "gpu"]`
- **Topology types:** `["slingshot", "nvlink"]`

**Usage:**
```bash
srun -N 4 --ntasks=16 --gpus-per-task=1 torchrun --nproc_per_node=4 --nnodes=4 \
    -m syssim.network.profiler --hierarchy-config examples/configs/perlmutter_mesh.json
```

### 2. `dgx_a100_mesh.json`
**Topology:** NVIDIA DGX A100
- **Mesh:** 2 nodes × 8 GPUs/node = 16 ranks
- **Dimensions:** `["node", "gpu"]`
- **Topology types:** `["infiniband", "nvlink"]`

**Usage:**
```bash
srun -N 2 --ntasks=16 --gpus-per-task=1 torchrun --nproc_per_node=8 --nnodes=2 \
    -m syssim.network.profiler --hierarchy-config examples/configs/dgx_a100_mesh.json
```

### 3. `default_profiling_params.json`
**Shared profiling parameters** used by all configs:
- Message sizes: 4KB to 2GB
- Runs per size: 10
- Adaptive sizing: lookahead=5, pfact=3.0

## Config Format

### Required Fields

```json
{
  "topology_name": "your_cluster_name",
  "description": "Human-readable description",
  "mesh": {
    "shape": [4, 4],                     // REQUIRED: Mesh dimensions
    "dimension_names": ["node", "gpu"],  // REQUIRED: Names for each dimension
    "ranks_order": "C"                   // Optional: 'C' (row-major, default) or 'F' (column-major)
  },
  "layers": [
    {
      "name": "layer_name",
      "topology_type": "nvlink",         // nvlink, infiniband, slingshot, ethernet, custom
      "scope": {                         // REQUIRED: Mesh scope specification
        "vary_dims": ["gpu"],            // Dimensions that vary
        "fix_dims": {"node": 0},         // Dimensions held constant (explicit int values)
        "num_pairs": 2                   // Optional: Number of rank pairs to profile (default 1)
      },
      "description": "Human-readable description",
      "expected_bandwidth_gbs": 300.0   // Optional: Bandwidth hint for validation
    }
  ],
  "profiling_params": {
    "min_size": 4096,
    "max_size": 2147483648,
    "num_runs": 10,
    "lookahead": 5,
    "pfact": 3.0
  }
}
```

### Mesh Scope Examples

**Intra-node (all GPUs on node 0):**
```json
{
  "vary_dims": ["gpu_in_node"],
  "fix_dims": {"node": 0}
}
// Result: ranks [0, 1, 2, 3] (for 4 GPUs/node)
```

**Inter-node (GPU 0 on all nodes):**
```json
{
  "vary_dims": ["node"],
  "fix_dims": {"gpu_in_node": 0}
}
// Result: ranks [0, 4, 8, 12] (for 4 GPUs/node)
```

**Inter-rack (node 0, GPU 0 on each rack):**
```json
{
  "vary_dims": ["rack"],
  "fix_dims": {"node_in_rack": 0, "gpu_in_node": 0}
}
// Result: ranks [0, 16] (for 2 racks, 4 nodes/rack, 4 GPUs/node)
```

## Creating New Configs

### Step 1: Determine Mesh Shape

Identify the logical dimensions of your cluster:
- **2D mesh:** `[num_nodes, gpus_per_node]`
- **3D mesh:** `[num_racks, nodes_per_rack, gpus_per_node]`
- **4D mesh:** `[num_datacenters, racks_per_dc, nodes_per_rack, gpus_per_node]`

**Example (DGX A100):**
- 2 nodes, 8 GPUs/node → `shape: [2, 8]`

### Step 2: Define Dimension Names

Choose descriptive names for each dimension:
- Common: `"node"`, `"gpu_in_node"`, `"rack"`, `"node_in_rack"`
- Must be unique

**Example:**
```json
{
  "shape": [2, 8],
  "dimension_names": ["node", "gpu_in_node"]
}
```

### Step 3: Specify Layer Scopes

For each network layer, specify which dimensions vary/stay constant:

**Intra-node (NVLink):**
```json
{
  "vary_dims": ["gpu_in_node"],  // Vary GPU
  "fix_dims": {"node": 0}        // Fix node=0
}
```

**Inter-node (InfiniBand):**
```json
{
  "vary_dims": ["node"],          // Vary node
  "fix_dims": {"gpu_in_node": 0}  // Fix GPU=0
}
```

### Step 4: Validate Config

Load and validate your config:
```python
from syssim.network.profiler import load_hierarchy_config

config = load_hierarchy_config("your_config.json")
config.validate()  # Raises ValueError if invalid

mesh = config.get_device_mesh()
print(f"Mesh shape: {mesh.shape}")
print(f"Total ranks: {mesh.total_ranks}")
```

## Migration from Old Configs

**Old format (NO LONGER SUPPORTED):**
```json
{
  "layers": [
    {
      "ranks": [0, 1, 2, 3],      // ❌ Explicit ranks removed
      "rank_pairs": [[0, 1]]      // ❌ Explicit pairs removed
    }
  ]
}
```

**New format (REQUIRED):**
```json
{
  "mesh": {                       // ✅ Mesh required
    "shape": [4, 4],
    "dimension_names": ["node", "gpu"]
  },
  "layers": [
    {
      "scope": {                  // ✅ Scope required
        "vary_dims": ["gpu"],
        "fix_dims": {"node": 0}
      }
    }
  ]
}
```

### Conversion Patterns

**Pattern 1: Intra-node**
```
ranks=[0,1,2,3] → vary_dims=["gpu"], fix_dims={"node": 0}
ranks=[4,5,6,7] → vary_dims=["gpu"], fix_dims={"node": 1}
```

**Pattern 2: Inter-node**
```
ranks=[0,4,8,12] → vary_dims=["node"], fix_dims={"gpu": 0}
```

**Pattern 3: All ranks**
```
ranks=[0..15] → vary_dims=["node", "gpu"], fix_dims={}
```

## Troubleshooting

### Error: "Missing 'mesh' field"
**Cause:** Config doesn't have `mesh` field (old format)
**Fix:** Add mesh specification (see format above)

### Error: "Missing 'scope' in layer"
**Cause:** Layer doesn't have `scope` field (old format)
**Fix:** Replace `ranks` and `rank_pairs` with `scope`

### Error: "fix_dims contains 'X' not in mesh.dimension_names"
**Cause:** Dimension name in `fix_dims` doesn't match `mesh.dimension_names`
**Fix:** Use exact dimension name from mesh (case-sensitive)

### Error: "fix_dims['X'] = Y out of bounds"
**Cause:** Fix dimension value exceeds mesh shape
**Fix:** Ensure `0 <= value < shape[dim_idx]`

### Error: "<2 ranks, cannot form pairs"
**Cause:** Mesh slice has only 1 rank
**Fix:** Check that vary_dims actually vary (not all fixed)

## See Also

- **DeviceMesh documentation:** `syssim/network/device_mesh.py`
- **Profiler documentation:** `syssim/network/profiler.py`
- **Implementation summary:** `logs/DEVICE_MESH_IMPLEMENTATION_SUMMARY.md`
- **Unit tests:** `tests/test_device_mesh.py`, `tests/test_mesh_config_loading.py`
