# SysSim Design Document

## 1. Design Overview

### 1.1 System Purpose

**syssim** is a PyTorch operator-level performance simulator that estimates neural network execution time without running actual computation. It traces model execution to build a computational graph (DAG), estimates per-operator runtime, and computes the critical path through multi-stream execution.

**Operator types**:
- **Compute operations** (GEMM, ATTN, MATH): Estimated using hybrid roofline model (analytical + ML efficiency predictor)
- **Collective operations** (allreduce, broadcast, etc.): Estimated using network simulator (topology + LogGP + congestion modeling)

### 1.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PyTorch Model                             │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Tracing Infrastructure                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ TorchDispatchMode + FakeTensorMode                        │  │
│  │  • Intercepts all PyTorch operations                      │  │
│  │  • Uses fake CUDA tensors (requires CUDA)                 │  │
│  │  • Two-phase: model conversion → execution trace          │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                            │
│  ┌──────────────────▼───────────────────────────────────────┐  │
│  │ Dependency Tracking                                       │  │
│  │  • TensorStorageTracker (data dependencies)              │  │
│  │  • CUDAEventTracker (cross-stream synchronization)       │  │
│  └──────────────────┬───────────────────────────────────────┘  │
└─────────────────────┼───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Operator Graph (DAG)                          │
│  • Nodes: Operators with estimated_time_ms                      │
│  • Edges: Data dependencies + stream dependencies               │
│  • Analysis: Critical path computation (multi-stream aware)     │
│  • Export: DOT (Graphviz), JSON, summary                        │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Runtime Estimation (called for each operator)            │  │
│  │                                                           │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │ Compute Operations (GEMM, ATTN, MATH)                │ │  │
│  │  │  Step 1: Roofline Model (Analytical)                 │ │  │
│  │  │   • Compute: FLOPs / peak_FLOP_s                     │ │  │
│  │  │   • Memory: bytes / peak_bandwidth                   │ │  │
│  │  │   • T_roofline = max(T_compute, T_memory)            │ │  │
│  │  │  Step 2: ML Efficiency Predictor                     │ │  │
│  │  │   • Features: dims + roofline + hardware             │ │  │
│  │  │   • ML: MLP or XGBoost                               │ │  │
│  │  │   • η = efficiency ∈ (0, 1]                          │ │  │
│  │  │  Step 3: Final Time                                  │ │  │
│  │  │   • T_actual = T_roofline / η                        │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │ Network Operations (allreduce, broadcast, etc.)      │ │  │
│  │  │  Step 1: Build Communication DAG                     │ │  │
│  │  │   • Algorithm → point-to-point operations            │ │  │
│  │  │   • Infer dependencies (same sender/receiver)        │ │  │
│  │  │  Step 2: Network Topology                            │ │  │
│  │  │   • Map (src, dst) to network resources              │ │  │
│  │  │   • 5 types: FullyConnected, Ring, Switch,           │ │  │
│  │  │     NVLinkMesh, Hierarchical                         │ │  │
│  │  │  Step 3: LogGP Performance Model                     │ │  │
│  │  │   • L, o, g, G parameters (analytical)               │ │  │
│  │  │   • Layer-specific for hierarchical                  │ │  │
│  │  │  Step 4: Event-Driven Simulation                     │ │  │
│  │  │   • Max-min fair bandwidth sharing                   │ │  │
│  │  │   • Congestion through resource contention           │ │  │
│  │  │   • Returns makespan and per-op timing               │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Core Components

#### Common Infrastructure

1. **Operator Graph IR** (`operator_graph.py`)
   - Common DAG representation for all operator types
   - Seven operator types: GEMM, ATTN, MATH (compute), COLLECTIVE (communication), MEMORY, BARRIER, STREAM_SYNC
   - Critical path algorithm with multi-stream support
   - Nodes store estimated_time_ms from either compute or network estimation

2. **Tracing Infrastructure** (`tracer.py`)
   - PyTorch dispatch interception using `TorchDispatchMode`
   - Fake CUDA tensors for GPU kernel dispatch (requires CUDA)
   - Dependency tracking through tensor storage and CUDA events

3. **Hardware Configuration** (`config.py`)
   - Hardware specifications (peak FLOP/s, memory bandwidth, network parameters)
   - Execution mode (TRAINING)
   - Simulator configuration

4. **Public API** (`api.py`)
   - `trace_model_for_training()` - Forward + backward pass

#### Compute Operation Estimation (`compute/`)

5. **Hybrid Performance Estimation**
   - Roofline model: Analytical performance ceiling
   - ML efficiency predictor: Learned correction factors
   - FLOP counting registry for operators
   - Used for: GEMM, ATTN, MATH operators

#### Network Operation Estimation (`network/`)

6. **Collective Communication Algorithms**
   - 8 collective primitives (allreduce, broadcast, reduce, scatter, gather, etc.)
   - Generates communication patterns as DAGs
   - Automatic dependency inference

7. **Network Topologies**
   - 5 topology models (FullyConnected, Ring, Switch, NVLink, Hierarchical)
   - Maps message routes to network resources
   - Supports multi-layer hierarchical networks

8. **LogGP Performance Model**
   - Analytical communication time model (L, o, g, G parameters)
   - Layer-specific parameters for hierarchical topologies
   - Protocol detection for different message sizes

9. **Event-Driven Simulation Engine**
   - Max-min fair bandwidth sharing
   - Congestion modeling through resource contention
   - Returns per-operation timing and makespan
   - Used for: COLLECTIVE operators

10. **Device Mesh Abstraction**
    - N-dimensional logical device layout
    - Auto-generates profiling layers from mesh dimensions
    - Hierarchical LogGP parameter profiling

---

## 2. Motivation and Concepts

### 2.1 The Performance Estimation Problem

**Challenge**: Estimate neural network execution time on specific hardware without running actual computation.

**Use cases**:
- Simulate model performance on unavailable hardware
- Optimize model architecture for target accelerators
- Predict training time before resource allocation
- Evaluate multi-stream execution strategies

**Requirements**:
1. **Accuracy**: Predictions within acceptable error margins
2. **Interpretability**: Explain why operations are slow (compute-bound vs memory-bound)
3. **Generalization**: Predict time for unseen operator sizes
4. **Hardware portability**: Adapt to new accelerators with minimal effort

### 2.2 Why Hybrid Model?

**Pure analytical models (roofline only)** have high prediction errors because they ignore:
- GPU utilization inefficiencies
- Cache hierarchy effects
- Kernel launch overhead (microsecond-scale per operation)
- Warp scheduling artifacts

**Pure machine learning models** can achieve better accuracy but:
- Require massive training datasets
- Lack interpretability (black box)
- Extrapolate poorly beyond training distribution
- Cannot explain performance bottlenecks

**Hybrid approach** combines strengths of both:
- **Roofline** provides physics-based scaling laws and bottleneck identification
- **Efficiency ML** learns real-world deviations from analytical ceiling
- **Result**: Good accuracy with interpretability and extrapolation

### 2.3 Core Equation

The hybrid model separates performance prediction into two components:

```
T_actual = T_roofline / η
```

Where:
- **T_roofline**: Analytical performance ceiling (multi-dimensional roofline model)
  - Deterministic, hardware-agnostic scaling laws
  - Computes maximum of compute and memory constraints

- **η**: Learned efficiency factor
  - Ratio of roofline time to actual time
  - Range: (0, 1], where 1.0 = perfect efficiency
  - Captures GPU utilization, cache effects, overhead, scheduling

- **T_actual**: Predicted execution time in milliseconds

### 2.4 Component Separation

#### Roofline Model (Analytical)
**Owns**: Scaling laws, hardware constraints, bottleneck identification

**Inputs**:
- Operation parameters (matrix dimensions, batch sizes, sequence lengths)
- Hardware specifications (peak FLOP/s, memory bandwidth)

**Outputs**:
- Multi-dimensional ceiling: maximum of all hardware constraints
- Dominant constraint: which resource limits performance

**Properties**:
- Deterministic and interpretable
- Generalizes across problem sizes
- Hardware-agnostic formula (hardware-specific parameters)

#### Efficiency Model (Machine Learning)
**Owns**: Real-world deviations from analytical ceiling

**Captures**:
- GPU core utilization patterns
- Cache hit/miss behavior
- Kernel launch overhead
- Warp scheduling inefficiencies

**Inputs**:
- Operation dimensions (log-scaled)
- Roofline envelope features (constraint times, ratios)
- Hardware descriptors (peak ratios, capacities)

**Outputs**:
- Efficiency η representing fraction of peak performance achieved

**Properties**:
- Hardware-specific (trained per GPU architecture)
- Learned from profiled data
- Efficiency scales with operation size

### 2.5 Design Benefits

1. **Interpretability**: Physics-based roofline provides transparent reasoning
   - Can explain: "This operation is memory-bound (bandwidth limited)"
   - Can quantify: "Compute constraint is tighter than memory constraint by X factor"

2. **Extrapolation**: ML learns corrections, not raw time
   - Roofline handles quadratic/cubic scaling
   - ML learns efficiency curves within each regime

3. **Moderate data requirements**: Thousands (not tens of thousands) of profiled configurations

4. **Hardware portability**: Roofline adapts automatically
   - Only efficiency models need retraining for new hardware

---

## 3. Architecture Details

### 3.1 Operator Graph IR

The **OperatorGraph** is the intermediate representation for traced execution, capturing operators, dependencies, and estimated times in a directed acyclic graph (DAG).

#### 3.1.1 OperatorType Enum

Seven operator types with distinct semantics:

- **GEMM**: Matrix multiply operations (mm, addmm, bmm, matmul, linear)
- **ATTN**: Scaled dot-product attention variants
- **MATH**: Generic math operations (layernorm, relu, softmax, etc.)
- **COLLECTIVE**: Distributed operations (all_reduce, all_gather, etc.)
- **MEMORY**: Data transfers (host↔device, device↔device)
- **BARRIER**: Wait for ALL streams (global synchronization)
- **STREAM_SYNC**: Wait for specific stream (targeted synchronization)

**Design decisions**:
- **GEMM vs ATTN**: Separate types because modern GPUs have different peak FLOP rates for these operations
- **BARRIER vs STREAM_SYNC**: Different critical path semantics (global vs targeted)
- **MATH**: Generic category for all other compute operations

#### 3.1.2 OperatorNode

Each node represents a single operator in the execution graph with:

- **Identity**: Unique node ID, operator type, operator name
- **Dependencies**: Data dependencies (producer node IDs), cross-stream dependencies
- **Metadata**: Input/output tensor shapes, CUDA stream ID
- **Performance**: Estimated execution time, operator-specific configuration
- **Timing**: Earliest start/finish times (computed during critical path analysis)

#### 3.1.3 Critical Path Algorithm

Computes total execution time accounting for multi-stream parallelism:

**Algorithm**:
1. Topological sort (Kahn's algorithm) for dependency order
2. Track earliest available time per stream
3. For each operator, compute earliest start based on:
   - Stream availability
   - Data dependency completion
   - Stream synchronization constraints

**Special handling**:
- **BARRIER**: Waits for ALL streams (maximum of all stream times)
- **STREAM_SYNC**: Waits for target stream only
- **Other operators**: Normal same-stream ordering with data dependencies

**Output**: Critical path time in milliseconds (maximum finish time across all streams)

### 3.2 Tracing Infrastructure

The tracing system intercepts PyTorch operations and builds the operator graph **without executing actual computation** using `FakeTensorMode`.

#### 3.2.1 Why Fake CUDA Tensors?

**Problem**: CPU tensors dispatch to different kernels than GPU tensors. For accurate GPU simulation, the tracer must observe GPU kernel dispatch patterns (e.g., flash attention, efficient attention, cuDNN attention variants).

**Solution**: Use fake CUDA tensors to get GPU dispatch without requiring actual hardware execution.

#### 3.2.2 Two-Phase Tracing

**Phase 1: Model Conversion to Fake CUDA Tensors** (tracer NOT active)
- Converts model parameters and buffers to fake CUDA tensors
- Must run outside tracer mode to avoid spurious operator nodes
- Directly mutates internal parameter/buffer dictionaries
- Creates restoration log for cleanup

**Phase 2: Trace Execution** (tracer active)
- Intercepts every PyTorch operation via `__torch_dispatch__`
- Classifies operators into types
- Extracts dependencies from tensor storage
- Estimates runtime using hybrid model
- Creates operator nodes and adds to graph
- Tracks output tensors for dependency chains
- Restores original model state on completion

#### 3.2.3 Dependency Tracking

**TensorStorageTracker**: Maps tensor storage pointers to producer operators.
- Registers outputs when operators execute
- Looks up producers when operators consume inputs
- Builds data dependency edges in the graph

**CUDAEventTracker**: Captures cross-stream synchronization.
- Monkey-patches `torch.cuda.Event.record()` to track event sources
- Monkey-patches `torch.cuda.Event.wait()` to create STREAM_SYNC nodes
- Enables accurate multi-stream execution modeling

#### 3.2.4 Operator Classification

Operators are classified into types based on:
- **Namespace**: Collective operations (c10d namespace)
- **Function identity**: Specific operators (matrix multiply, attention, memory copy)
- **Default category**: Everything else maps to MATH

Classification determines:
- Which peak FLOP rate to use (GEMM/ATTN use tensor unit peak)
- How to count FLOPs
- How to extract operation parameters

### 3.3 Configuration System

#### 3.3.1 ExecutionMode

The execution mode affects performance prediction:

- **TRAINING**: Forward + backward passes

**Impact on prediction**:
- TRAINING traces both forward and backward passes through the model

#### 3.3.2 HardwareInfo

Encapsulates hardware specifications:

- **peak_tflops_mm**: Peak tensor unit throughput (TFLOP/s)
- **peak_tflops_math**: Peak vector unit throughput (TFLOP/s)
- **peak_memory_bandwidth_gbps**: Memory bandwidth (GB/s)
- **peak_tflops_mm_conservative**: Conservative peak for small operations

**Unit definitions**:
- Peak FLOP rates: TFLOP/s (10^12 FLOP/s)
- Memory bandwidth: GB/s (10^9 bytes/s)

**Peak selection**: Large operations use peak tensor unit throughput, small operations use conservative peak to account for kernel launch overhead.

#### 3.3.3 SimulatorConfig

Combines hardware information with execution parameters:
- Hardware specifications (HardwareInfo)

### 3.4 Public API

#### 3.4.1 trace_model_for_training()

Traces forward and backward passes for training scenarios.

**Inputs**:
- PyTorch model
- Example inputs
- Simulator configuration
- Optional loss function (default: sum reduction)

**Process**:
1. Validates CUDA availability
2. Creates tracer with configuration
3. Converts model to fake CUDA tensors
4. Traces forward pass
5. Applies loss function
6. Traces backward pass via automatic differentiation
7. Restores original model state

**Output**: OperatorGraph with forward + backward execution

### 3.5 Roofline Model Component

The roofline model implements a **multi-dimensional performance ceiling** based on hardware constraints.

#### 3.5.1 Mathematical Foundation

The roofline ceiling is the maximum of all hardware constraints:

```
T_roofline = max(T_compute, T_memory)
```

**Compute Constraint** (FLOP-bound):
- Time limited by computational throughput
- Formula: FLOPs / peak_FLOP_s
- Depends on operation FLOP count and hardware peak

**Memory Constraint** (Bandwidth-bound):
- Time limited by data transfer bandwidth
- Formula: bytes_transferred / peak_bandwidth
- Depends on tensor sizes and memory bandwidth

**Bottleneck Identification**:
- Compute-bound: Compute constraint dominates
- Memory-bound: Memory constraint dominates
- Constraint ratio quantifies bottleneck distance

#### 3.5.2 Size-Aware Peak Selection

Operations achieve different fractions of peak performance based on size:

- **Large operations** (all dimensions ≥ 512): Use peak tensor unit throughput
  - Good GPU utilization
  - Amortized launch overhead

- **Small operations** (any dimension < 512): Use conservative peak
  - Limited parallelism
  - Launch overhead dominates (~7 microseconds)

#### 3.5.3 FLOP Counting Registry

Operator-specific FLOP formulas registered for:
- Matrix multiply: 2 × M × N × K (multiply + accumulate)
- Attention: 2 × batch × heads × seq_len² × head_dim
- Other operations: Element-wise, reduction, normalization formulas

#### 3.5.4 Memory Transfer Estimation

Accounts for:
- Input tensor reads
- Output tensor writes
- PyTorch memory allocator alignment (512-byte minimum)

#### 3.5.5 Unit System

**Input units** (from user):
- Peak FLOP rates: TFLOP/s (tera = 10^12)
- Memory bandwidth: GB/s (giga = 10^9)

**Internal calculations**: Nanoseconds for numerical precision

**Output units** (to user):
- All estimated times in milliseconds (ms)
- Critical path time in milliseconds

### 3.6 Efficiency Model Component

The efficiency η represents the **fraction of peak performance achieved**:

```
η = T_roofline / T_measured ∈ (0, 1]
```

**Interpretation**:
- η = 1.0: Perfect efficiency (achieves roofline ceiling)
- η = 0.3: 30% efficient (actual time is 3.3× roofline)
- η < 0.1: Very inefficient (overhead or poor utilization dominates)

**What efficiency captures**:
1. GPU core utilization patterns
2. Cache hierarchy effects (L1/L2 hit rates)
3. Kernel launch overhead (fixed cost per operation)
4. Warp scheduling and memory latency hiding
5. Precision-specific throughput characteristics

**Efficiency scaling**: Varies with operation size
- Tiny operations: Very low efficiency (overhead dominates)
- Small operations: Low efficiency (partial utilization)
- Large operations: Moderate to good efficiency (limited by cache/memory)

#### 3.6.1 Feature Engineering

Efficiency models use combined features from three categories:

**Base Features** (log-scaled dimensions):
- GEMM: Matrix dimensions (M, N, K), aspect ratios, extremes
- Attention: Batch, heads, sequence length, head dimension, products
- Math: Batch, hidden size, total size

**Roofline Envelope Features** (bottleneck characteristics):
- Constraint times (log-scaled)
- Constraint ratio (bottleneck distance)
- Dominant constraint (one-hot encoding)

**Hardware Descriptors**:
- Capacity ratios (architectural characteristics)
- Absolute capacities (scale information)

**Why roofline envelope features matter**:
- Encode compute vs memory-bound regimes
- Correlate with efficiency patterns
- Provide scaling behavior information

#### 3.6.2 ML Architectures

Two backend implementations are available:

**MLP Backend**: Multi-layer perceptron with:
- 4 hidden layers with batch normalization
- Dropout regularization
- Sigmoid output (constrains to valid efficiency range)

**XGBoost Backend**: Gradient boosted trees with:
- Histogram-based tree construction
- Regularization (L1/L2, subsampling)
- Early stopping during training

Both backends:
- Take same feature inputs
- Predict efficiency η ∈ (0, 1]
- Are hardware-specific (trained per GPU architecture)

#### 3.6.3 Model Management

**BackendManager** (singleton pattern):
- Loads efficiency models per operator type
- Environment-based model directory configuration
- Lazy loading for efficiency

### 3.7 Runtime Integration

#### 3.7.1 Execution Flow

For each traced operator:

1. **Roofline Estimation**:
   - Count FLOPs from operation parameters
   - Estimate bytes transferred
   - Compute compute and memory constraints
   - Return maximum constraint time and bottleneck

2. **Efficiency Prediction**:
   - Extract base features from operation dimensions
   - Extract roofline envelope features
   - Extract hardware descriptor features
   - Load appropriate ML model
   - Predict efficiency η

3. **Hybrid Estimate**:
   - Combine: T_actual = T_roofline / η
   - Clip efficiency to valid range for stability
   - Return estimated time in milliseconds

#### 3.7.2 Fallback Behavior

Graceful degradation when models unavailable:

**Scenarios**:
- Model directory not configured
- Model file missing for operator type
- Prediction exception during model prediction

**Fallback strategy**:
- Return efficiency = 1.0 (pure roofline estimate)
- Log warning about missing model
- Continue tracing with conservative estimates

**Result**: Conservative but interpretable predictions using roofline only

---

## 4. Hardware Adaptation

### 4.1 Why Hardware-Specific ML Models?

**Critical insight**: ML efficiency predictors are **NOT portable** across GPU architectures.

**Root causes**:

1. **Cache Hierarchies Differ**:
   - Different cache sizes per streaming multiprocessor
   - Different total L2 cache capacities
   - Affects efficiency curves for different operation sizes

2. **Kernel Launch Overhead Varies**:
   - Different fixed overhead per kernel launch
   - Small operations dominated by this overhead
   - Overhead magnitude varies across GPU generations

3. **Scheduling Characteristics**:
   - Warp scheduler architecture varies
   - SM occupancy limits differ
   - Memory latency hiding effectiveness changes

4. **Precision Support**:
   - Different tensor unit capabilities
   - Different peak FLOP rates for various precisions
   - Some architectures lack tensor units entirely

**Consequence**: ML efficiency predictors trained on one architecture have poor accuracy on another. Hardware-specific training is required.

### 4.2 What Needs to Change for New Hardware

#### 4.2.1 Roofline Model (Analytical Component)

**What changes**: Hardware parameters only

**Required inputs**:
- Peak tensor unit FLOP rate (TFLOP/s)
- Peak vector core FLOP rate (TFLOP/s)
- Memory bandwidth (GB/s)
- Conservative peak for small operations (optional)

**How to obtain**:
1. **PyTorch auto-detection** (recommended):
   - Use PyTorch utilities to query device capabilities
   - Convert units appropriately (PFLOP/s to TFLOP/s)
   - Query memory bandwidth from device properties

2. **Hardware database lookup** (fallback):
   - Vendor specifications from data sheets
   - Documented peak performance numbers
   - Reference benchmarks

3. **Empirical measurement** (most accurate):
   - Micro-benchmarks for peak FLOP/s
   - Memory bandwidth tests
   - Validate against vendor specifications

**What stays the same**:
- Roofline formula and constraint logic
- FLOP counting methodology
- Memory transfer estimation
- Size-based peak selection approach

#### 4.2.2 ML Efficiency Predictor

**What changes**: Trained ML model weights (must re-profile)

**Required steps**:
1. **Profile on target hardware**: Run profiler to collect actual execution times
2. **Train ML models**: Fit MLP or XGBoost models to profiled data
3. **Validate accuracy**: Check prediction error on held-out data
4. **Deploy**: Save trained ML models for runtime loading

**Profiling scope**:
- All operator types (GEMM, attention, math)
- Wide range of sizes (small to large)
- Sufficient runs for statistical stability

**What stays the same**:
- Feature engineering (same features extracted)
- ML architecture (MLP or XGBoost structure)
- Training methodology
- Feature extraction logic

### 4.3 When to Re-Profile

**Triggers for re-profiling**:

1. **New GPU Architecture**:
   - Different cache hierarchy
   - Different SM count or memory controller
   - Example: Migrating from Hopper to next-generation architecture

2. **Major CUDA/Driver Updates**:
   - Kernel implementations change
   - New optimizations introduced
   - Typically: CUDA major version updates

3. **Roofline Formula Changes**:
   - If analytical roofline model updated
   - Must retrain ML efficiency predictors to match new roofline

4. **Observed Accuracy Degradation**:
   - Monitor prediction error on validation data
   - Significant error increase indicates need for re-profiling

**Profiling cost**:
- Time: Minutes per operator type on CUDA hardware
- Compute: Thousands of configurations × multiple runs each
- Storage: Modest (tens of megabytes per operator type)

---

## 5. Network Communication Simulator

The network simulator models collective communication operations for distributed training workloads. It simulates message-passing execution on multi-GPU clusters with realistic congestion and bandwidth contention.

### 5.1 Core Components

**Simulation Engine**:
- Event-driven simulation with max-min fair bandwidth sharing
- Models congestion naturally through resource sharing
- Each operation's bandwidth = min(resource_bandwidth / num_users) across its path
- Returns per-operation timing and total makespan

**Network Topologies**:
- **FullyConnectedTopology**: Dedicated link per pair (no contention, validation baseline)
- **RingTopology**: Bidirectional ring with shortest-path routing
- **SwitchTopology**: Star topology with shared switch bottleneck
- **NVLinkMeshTopology**: Fully-connected NVLink mesh (intra-node)
- **HierarchicalTopology**: Multi-layer (NVLink intra-node + InfiniBand/Slingshot inter-node)

**Collective Communication Algorithms**:
- **allreduce**: Ring algorithm (reduce-scatter + allgather)
- **broadcast**: Binomial tree from root
- **reduce**: Binomial tree to root
- **reduce_scatter**: First half of ring allreduce
- **allgather**: Second half of ring allreduce
- **alltoall**: Direct with staggered pairings
- **scatter**: Flat from root (serialized)
- **gather**: Flat to root (parallel)

**LogGP Performance Model**:
- Analytical model for communication time estimation
- **L**: Latency (seconds) - one-way message delay
- **o**: CPU overhead (seconds) - per-message processing time
- **g**: Gap per message (seconds) - minimum interval between consecutive messages
- **G**: Gap per byte (seconds/byte) - inverse of bandwidth
- Supports layer-specific parameters for hierarchical topologies

### 5.2 DAG Construction and Dependencies

**DAG Builder**:
- Converts high-level communication patterns into operation DAGs
- Each operation represents a single point-to-point message with sender, receiver, size
- Dependency rules infer ordering automatically:
  - **Rule 1**: Same sender → sequential (sender can't overlap sends)
  - **Rule 2**: Same receiver → sequential (receiver can't overlap receives)
  - Independent operations execute in parallel

**Operation Structure**:
- **Static properties**: sender, receiver, size, layer (topology type)
- **Dependencies**: List of predecessor operations that must complete first
- **Runtime state**: start_time, finish_time, remaining_bytes (populated by simulator)

### 5.3 Hierarchical Profiling with Device Mesh

**Device Mesh Abstraction**:
- N-dimensional logical device layout (e.g., [nodes, GPUs/node] or [racks, nodes, GPUs])
- Maps coordinates to global ranks using row-major or column-major ordering
- Auto-generates profiling layers from mesh dimensions

**Mesh Configuration**:
```python
{
  "topology_name": "perlmutter",
  "mesh": {
    "shape": [4, 4],                      # 4 nodes × 4 GPUs/node
    "dimension_names": ["node", "gpu"],   # Dimension labels
    "topology_types": ["slingshot", "nvlink"]  # One type per dimension
  }
}
```

**Auto-Generated Profiling Layers**:
- System creates one profiling layer per mesh dimension
- Each layer varies that dimension while fixing all others to 0
- Example: `["node", "gpu"]` generates:
  - Layer "node": varies nodes (GPU fixed to 0) → profiles inter-node (Slingshot)
  - Layer "gpu": varies GPUs (node fixed to 0) → profiles intra-node (NVLink)

**LogGP Parameter Profiler**:
- Executes ping-pong microbenchmarks across message sizes
- Detects protocol changes (eager vs rendezvous) via least-squares deviation
- Extracts L, o, g, G parameters using PRTT (Parametrized Round Trip Time) method
- Outputs layer-specific LogGP parameters for simulation

### 5.4 Simulation Workflow

**Typical workflow**:

```python
from syssim.network import allreduce

ops = allreduce(ranks=list(range(8)), total_size=1e9)
result = simulate(ops, topo, loggp)
print(f"Makespan: {result.makespan * 1e3:.2f} ms")
```

**Simulation outputs**:
- Per-operation start and finish times
- Total execution time (makespan)
- Per-rank finish times
- Bottleneck identification through resource usage

### 5.5 Max-Min Fair Bandwidth Sharing

**Congestion modeling**:
- Resources are shared among all active operations using them
- Each operation gets fair share of the most congested resource on its path
- Effective bandwidth for operation i:
  ```
  bandwidth_i = min(resource_bandwidth_r / num_ops_sharing_r) for all r on path_i
  ```

**Example**:
- 3 operations share a 100 GB/s switch
- Each gets 33.3 GB/s through the switch
- If one operation also uses a 25 GB/s link, it's limited to 25 GB/s (bottleneck)

**Benefits**:
- Natural congestion modeling without explicit contention rules
- Works for arbitrary topologies
- Realistic multi-operation overlap behavior

---

## 6. Diffusion Model Support

### 6.1 Motivation

Diffusion models (Stable Diffusion, SDXL, Wan2.2, etc.) have fundamentally different execution patterns from autoregressive LLMs:

| Characteristic | LLM Inference | Diffusion Inference |
|---------------|---------------|-------------------|
| Core loop | Autoregressive token generation | Iterative denoising (fixed steps) |
| Per-step shape | Varies (prefill=full seq, decode=1 token) | Constant (same latent shape every step) |
| Components | Single model | Pipeline: text encoder → denoiser → VAE decoder |
| KV cache | Yes (grows with sequence) | No |
| Classifier-free guidance | N/A | 2× forward passes per step when guidance > 1.0 |
| Dominant cost | Attention (memory-bound at decode) | MatMul/Attention in denoiser × num_steps |

These differences mean that a single `trace_model_for_inference()` call is insufficient — the simulator must understand the pipeline decomposition and iterative nature of diffusion.

### 6.2 Pipeline Decomposition

A diffusion inference pipeline consists of three sequential stages:

```
┌──────────────┐     ┌──────────────────────────┐     ┌──────────────┐
│ Text Encoder │     │     Denoiser (DiT/UNet)   │     │ VAE Decoder  │
│   (1× run)   │ ──▶ │  (N steps × CFG passes)  │ ──▶ │   (1× run)   │
│  e.g. T5/CLIP│     │  e.g. DiT, UNet           │     │  latent→pixel│
└──────────────┘     └──────────────────────────┘     └──────────────┘
```

**Total pipeline time**:
```
T_pipeline = T_text_encoder + T_denoise_step × num_steps × cfg_multiplier + T_vae_decoder
```

Where:
- **T_text_encoder**: One-time text encoding cost (typically small)
- **T_denoise_step**: Single denoising step cost (dominant term)
- **num_steps**: Number of denoising iterations (e.g. 20, 50)
- **cfg_multiplier**: 2 if classifier-free guidance is active (guidance_scale > 1.0), else 1
- **T_vae_decoder**: One-time latent-to-pixel decoding cost

**Key insight — trace once, multiply by N**: Every denoising step operates on identical tensor shapes (the latent remains the same size throughout the diffusion process). Therefore the operator graph for step 1 is structurally identical to step 50. We trace a single step and multiply by `num_steps × cfg_multiplier` to get the total denoising cost.

### 6.3 Architecture

#### 6.3.1 Configuration

**DiffusionConfig** (`config.py`):
```python
@dataclass
class DiffusionConfig:
    num_inference_steps: int = 50      # Denoising iterations
    guidance_scale: float = 7.5        # CFG scale (>1.0 doubles passes)
    num_frames: int = 1                # Video frames (1 for image models)

    @property
    def cfg_multiplier(self) -> int:
        return 2 if self.guidance_scale > 1.0 else 1
```

**ExecutionMode.DIFFUSION_DENOISE** (`config.py`):
A new execution mode for single denoising step tracing. This mode:
- Runs forward-only (no backward pass)
- Does NOT activate KV-cache decode path (no cache_seq_len adjustment)
- Uses the standard roofline path, same as PREFILL

The mode exists as a semantic marker — it allows future specialization (e.g. diffusion-specific efficiency models, timestep-aware estimation) without breaking the existing code path.

#### 6.3.2 API

**trace_diffusion_pipeline()** (`api.py`):

The primary API for diffusion model simulation. Traces each pipeline component independently:

```python
def trace_diffusion_pipeline(
    denoise_model: nn.Module,          # DiT, UNet, or any denoiser
    denoise_inputs: Any,               # (latent, timestep, text_emb)
    config: SimulatorConfig,
    diffusion_config: DiffusionConfig | None = None,
    text_encoder: nn.Module | None = None,
    text_encoder_inputs: Any = None,
    vae_decoder: nn.Module | None = None,
    vae_decoder_inputs: Any = None,
) -> DiffusionPipelineResult:
```

**Execution flow**:

1. **Trace denoiser** (required): Creates `OperatorGraphTracer` with `DIFFUSION_DENOISE` mode, traces one forward pass of the denoising model.
2. **Trace text encoder** (optional): Creates separate tracer with `PREFILL` mode, traces text encoding.
3. **Trace VAE decoder** (optional): Creates separate tracer with `PREFILL` mode, traces VAE decoding.
4. **Compute aggregate time**: Combines per-stage critical paths with the multiplication formula.

Each component is traced with an independent `OperatorGraphTracer` instance — this ensures clean dependency graphs without cross-component aliasing.

**DiffusionPipelineResult** (`api.py`):

```python
@dataclass
class DiffusionPipelineResult:
    denoise_step_graph: OperatorGraph   # Graph for ONE denoising step
    text_encoder_graph: OperatorGraph | None
    vae_decoder_graph: OperatorGraph | None
    diffusion_config: DiffusionConfig
    denoise_step_ms: float              # Critical path of one step
    text_encoder_ms: float
    vae_decoder_ms: float
    total_pipeline_ms: float            # Aggregate pipeline time
```

The result preserves per-stage operator graphs for detailed analysis (operator breakdown, bottleneck identification) while also providing the aggregate pipeline timing.

#### 6.3.3 Diffusers Integration

**trace_diffusers_pipeline()** (`integrations/diffusers.py`):

Convenience wrapper that extracts components from a HuggingFace Diffusers `DiffusionPipeline`:

1. **Text encoder**: `pipeline.text_encoder` (if present)
2. **Denoiser**: `pipeline.transformer` (DiT-based) or `pipeline.unet` (UNet-based)
3. **VAE decoder**: `pipeline.vae.decoder` (if present)

The integration handles:
- Automatic latent shape calculation from pixel dimensions and VAE scale factor
- Synthetic input construction for each component
- Cross-attention dimension extraction from model config

**build_wan2_2_inputs()** (`integrations/diffusers.py`):

Helper that constructs denoiser inputs matching Wan2.2's architecture:
- 5D video latent: `(B, 16, T//4, H//8, W//8)` — 16 latent channels, 4× temporal and 8× spatial compression
- Timestep: scalar tensor
- Text embeddings: `(B, seq_len, 4096)` — UMT5-XXL hidden dimension

### 6.4 Wan2.2 as Reference Implementation

Wan2.2 is an Alibaba video generation model that serves as the reference for video diffusion support.

#### 6.4.1 Architecture Overview

```
Prompt ──▶ UMT5-XXL ──▶ [B, 512, 4096] text embeddings
                              │
Noise  ──▶ 3D DiT ◀──────────┘ (cross-attention)
           │  30 layers, hidden=1536, 24 heads
           │  3D patch embedding (1×2×2)
           │  Self-attn + Cross-attn + FFN per block
           ▼
Denoised latent ──▶ 3D VAE Decoder ──▶ Video frames
```

**Key architectural parameters** (Wan2.2-1.3B):

| Parameter | Value |
|-----------|-------|
| Latent channels | 16 |
| VAE spatial compression | 8× |
| VAE temporal compression | 4× |
| DiT hidden size | 1536 |
| DiT layers | 30 |
| DiT attention heads | 24 |
| Patch size | (1, 2, 2) — temporal × height × width |
| Text encoder | UMT5-XXL (cross_attention_dim=4096) |
| Default resolution | 480×832, 81 frames (5s @ 16fps + 1) |

#### 6.4.2 Latent Shape Calculation

For a video of `(H, W, F)` pixels/frames:
```
latent_T = ceil(F / temporal_compression)  = ceil(81 / 4) = 21
latent_H = H / spatial_compression         = 480 / 8      = 60
latent_W = W / spatial_compression         = 832 / 8      = 104
```

After patchification `(pt=1, ph=2, pw=2)`:
```
num_patches = (latent_T / pt) × (latent_H / ph) × (latent_W / pw)
            = 21 × 30 × 52 = 32,760 tokens
```

This 32K-token sequence is what the DiT transformer operates on — making self-attention the dominant compute cost.

#### 6.4.3 Compute Profile

For a single denoising step with the sequence above:
- **Self-attention**: O(seq² × hidden) per layer → 32,760² × 1,536 × 30 layers
- **Cross-attention**: O(seq × text_seq × hidden) per layer → 32,760 × 512 × 1,536 × 30 layers
- **FFN**: O(seq × hidden × 4 × hidden) per layer → 32,760 × 1,536 × 6,144 × 30 layers

With 50 steps and CFG (2× per step), the denoiser runs **100 times** — this is why per-step optimization matters.

### 6.5 Design Decisions

#### 6.5.1 Why Arithmetic Composition Instead of Graph Merging?

The pipeline stages are sequential and independent — there is no data dependency between step 1's graph and step 2's graph (they process different noise levels on the same shape). Merging N identical graphs would:
1. Create a massive graph (100× nodes) with no analytical benefit
2. Make critical path computation slower without improving accuracy
3. Lose the clear per-stage breakdown

Arithmetic composition (`step_time × N`) is exact for identical-shape iterations.

#### 6.5.2 Why a Separate ExecutionMode?

`DIFFUSION_DENOISE` currently follows the same code path as `PREFILL`. The separate mode exists for:
1. **Semantic clarity**: A denoising step is not a "prefill" — the intent is different
2. **Future extensibility**: Timestep-aware efficiency models, diffusion-specific operator patterns (e.g. AdaLN modulation), video-specific optimizations
3. **Profiling**: Allows collecting diffusion-specific profiling data tagged by mode

#### 6.5.3 Why Component-Level Tracing?

Each pipeline component is traced with a separate `OperatorGraphTracer`:
1. **Isolation**: No false dependencies between text encoder and denoiser tensors
2. **Modularity**: Users can trace only the denoiser (dominant cost) without needing the full pipeline
3. **Flexibility**: Components can be swapped (e.g. different text encoders) without re-tracing everything

### 6.6 Comparison with LLM Tracing

```
                    LLM                              Diffusion
                   ┌──────┐                         ┌──────────────────┐
  API layer        │trace_│ → OperatorGraph         │trace_diffusion_  │ → DiffusionPipelineResult
                   │model │                         │pipeline          │   (3 graphs + arithmetic)
                   └──┬───┘                         └──┬───┬───┬──────┘
                      │                                │   │   │
  Tracer layer     trace()                         trace() trace() trace()  ← same tracer
                      │                                │   │   │
  Roofline layer   estimate_runtime()              estimate_runtime()       ← same estimator
                   (DECODE has special path)        (no special path)
```

**What is shared** (zero code duplication):
- `OperatorGraphTracer` — same dispatch interception and dependency tracking
- `estimate_runtime()` — same roofline + efficiency pipeline
- `OperatorGraph` — same DAG IR and critical path analysis
- FLOP counting — same registry (convolution formulas already supported)

**What is new**:
- `DiffusionConfig` — steps, guidance scale, video frames
- `trace_diffusion_pipeline()` — multi-component orchestration + arithmetic composition
- `DiffusionPipelineResult` — multi-graph result with aggregate timing
- `integrations/diffusers.py` — HuggingFace Diffusers pipeline decomposition

### 6.7 Supported Diffusion Architectures

The implementation is architecture-agnostic — any PyTorch `nn.Module` can be traced as the denoiser. Tested patterns:

| Architecture | Pipeline Attribute | Example Models |
|-------------|-------------------|----------------|
| DiT (Diffusion Transformer) | `pipeline.transformer` | Wan2.2, PixArt, SD3, Flux |
| UNet | `pipeline.unet` | Stable Diffusion 1.x/2.x, SDXL |
| Any nn.Module | Direct `trace_diffusion_pipeline()` | Custom architectures |

The `_get_denoiser()` helper in the Diffusers integration auto-detects whether the pipeline uses a transformer or UNet backbone.

### 6.8 Limitations and Future Work

1. **Scheduler overhead not modeled**: The noise scheduler (DDPM, DDIM, Euler, etc.) performs lightweight tensor operations between denoising steps. These are negligible compared to the model forward pass but are not currently traced.

2. **VAE encoder not traced**: For image/video-to-image/video pipelines (e.g. img2img, inpainting), the VAE encoder is also needed. Currently only the decoder is supported.

3. **LoRA/IP-Adapter not modeled**: Adapter-based customizations modify the effective compute but are not explicitly supported.

4. **Multi-GPU diffusion**: Sequence parallelism and tensor parallelism for large DiT models (e.g. Wan2.2-14B) are not yet integrated with the network simulator.

5. **Timestep-dependent efficiency**: Real diffusion models may exhibit timestep-dependent performance patterns (e.g. early steps may trigger different attention patterns). The current model assumes uniform per-step cost.

---

### 5.6 LogGP Parameter Management

**LogGP Parameter Loader**:
- Loads profiled LogGP parameters from persistent storage
- Supports single-layer and multi-layer (hierarchical) configurations
- Protocol detection: selects appropriate parameters based on message size
- Validation: compares profiled vs expected bandwidth

**LogGP Parameter selection**:
```python
from syssim.network import load_loggp_params, get_protocol_for_size

# Load profiled LogGP parameters for specific network type
params = load_loggp_params("nvlink")

# Select protocol parameters based on message size
loggp = get_protocol_for_size(params, size=1e6)
```
