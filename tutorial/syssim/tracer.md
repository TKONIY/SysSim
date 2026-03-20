# `tracer.py` — PyTorch 模型追踪引擎

## 文件概述

`tracer.py` 是 SysSim 中最复杂也是最核心的模块，负责将 PyTorch 模型的执行过程转化为 `OperatorGraph`。它利用 PyTorch 的 `TorchDispatchMode` + `FakeTensorMode` 机制，在不执行真实 GPU 计算的情况下拦截所有算子调用，记录算子类型、张量形状、数据依赖关系，并估算每个算子的执行时间。

核心机制包括：
- **FakeTensor 模式**：使用"假"CUDA 张量进行形状推断，避免实际 GPU 内存分配和计算。
- **TorchDispatch 拦截**：在 PyTorch 的 dispatch 层拦截每个算子调用。
- **Storage 追踪**：通过追踪张量底层存储（untyped_storage）来正确处理视图（view）操作的别名关系。
- **CUDA Event 追踪**：通过 monkey-patch `torch.cuda.Event` 的 `record`/`wait` 方法来捕捉跨流同步依赖。

## 关键代码解析

### FakeTensor 转换

```python
def _to_fake_device(
    tensor: torch.Tensor,
    fake_mode: FakeTensorMode,
    device: str,
) -> torch.Tensor:
    meta = tensor.to(device="meta")
    return _FakeTensor(fake_mode, meta, torch.device(device))
```

将真实张量转为 FakeTensor 的过程分两步：
1. 先转为 `meta` 设备上的张量（只有形状信息，无数据）。
2. 包装为 `FakeTensor`，声称位于指定的 CUDA 设备上。

这样 PyTorch 的 dispatch 机制会将算子路由到 GPU kernel 变体（如 flash attention），但实际不会执行任何 GPU 计算。

### 模型参数转换与恢复

```python
def _convert_model_to_fake(
    model: nn.Module,
    fake_mode: FakeTensorMode,
    device: str,
) -> list[tuple[nn.Module, str, dict[str, torch.Tensor], bool]]:
    restore_log = []
    for mod in model.modules():
        # 转换 parameters
        orig_params = dict(mod._parameters)
        new_params = {}
        for k, v in orig_params.items():
            if v is not None:
                fake = _to_fake_device(v.data, fake_mode, device)
                if v.requires_grad:
                    fake.requires_grad_(True)
                new_params[k] = fake
        # 保存原始参数以便恢复
        restore_log.append((mod, "_parameters", orig_params, True))
        mod._parameters = new_params

        # 类似地转换 buffers...
    return restore_log
```

此函数遍历模型的每个子模块，将所有 `parameters` 和 `buffers` 替换为 FakeTensor。关键细节：
- 保留 `requires_grad` 属性，确保反向传播图能正确构建。
- 返回 `restore_log`，追踪完成后调用 `_restore_model()` 恢复原始参数，确保追踪过程不会破坏用户的模型。

### Storage 追踪器

```python
class TensorStorageTracker:
    def __init__(self) -> None:
        self._storage_to_producer: dict[int, str] = {}

    def register_output(self, tensor: torch.Tensor, producer_name: str) -> None:
        key = _storage_key(tensor)  # id(tensor.untyped_storage())
        self._storage_to_producer[key] = producer_name

    def get_producer(self, tensor: torch.Tensor) -> Optional[str]:
        key = _storage_key(tensor)
        return self._storage_to_producer.get(key)

    def register_alias(self, alias: torch.Tensor, source: torch.Tensor) -> None:
        src_key = _storage_key(source)
        producer = self._storage_to_producer.get(src_key)
        if producer is not None:
            alias_key = _storage_key(alias)
            self._storage_to_producer[alias_key] = producer
```

`TensorStorageTracker` 是依赖追踪的基础。它通过 `id(tensor.untyped_storage())` 将每个张量的底层存储映射到其生产者算子名称。

**为什么用 storage 而不是 tensor id？** 因为 PyTorch 的视图操作（如 `reshape`、`transpose`、`slice`）会创建新的 Tensor 对象，但共享同一个底层存储。通过追踪 storage id，视图操作的输出能正确地指向原始数据的生产者，从而建立准确的数据依赖关系。

`register_alias` 是一个安全网：在某些极端情况下视图操作可能创建新的 storage 对象，此方法将新 storage 链接到源 storage 的生产者。

### CUDA Event 追踪器

```python
class CUDAEventTracker:
    def install_hooks(self) -> None:
        tracker = self

        def patched_record(event_self, stream=None):
            stream_id = 0 if stream is None else getattr(stream, "stream_id", 0)
            event_id = id(event_self)
            tracker._event_to_stream[event_id] = stream_id
            last_op = tracker._last_op_on_stream.get(stream_id)
            if last_op is not None:
                tracker._event_to_last_op[event_id] = last_op

        def patched_wait(event_self, stream=None):
            # 创建 STREAM_SYNC 节点
            node = OperatorNode(
                name=f"stream_sync_{idx}",
                op_type=OperatorType.STREAM_SYNC,
                stream_id=stream_id,
                config={"target_stream": src_stream},
            )
            # 添加依赖...
            tracker._graph.add_operator(node)

        torch.cuda.Event.record = patched_record
        torch.cuda.Event.wait = patched_wait
```

通过 monkey-patch 替换 `torch.cuda.Event` 的 `record` 和 `wait` 方法：
- `record`：记录事件发生在哪个 CUDA 流，以及该流上最后一个算子是什么。
- `wait`：当某个流等待另一个流的事件时，创建一个 `STREAM_SYNC` 节点，建立跨流依赖关系。

这种机制让追踪器能够捕获模型中复杂的多流并行模式（如 Megatron-LM 中计算与通信的重叠）。

### 算子分类系统

```python
def _classify_op(
    func_packet: Any, func: Any, args: tuple, kwargs: dict
) -> tuple[Optional[OperatorType], dict[str, Any]]:
    func_name = str(func_packet)

    # 集合通信
    if "c10d" in func_name:
        return OperatorType.COLLECTIVE, {}

    # 跨设备拷贝
    copy_type = _is_cross_device_copy(func_packet, args, kwargs)
    if copy_type is not None:
        return copy_type, config

    # 矩阵乘法
    if func_packet in _GEMM_OPS:
        return OperatorType.GEMM, _extract_gemm_config(args)

    # 注意力计算
    if func_packet in _ATTN_OPS:
        return OperatorType.ATTN, _extract_attention_config(args)

    # 默认: 通用数学运算
    return OperatorType.MATH, {}
```

分类优先级从高到低：集合通信 > 跨设备拷贝 > GEMM > 注意力 > 通用数学运算。

对于 GEMM 和 ATTN 算子，还会提取额外的配置信息：

```python
def _extract_gemm_config(args: tuple) -> dict[str, Any]:
    # 从 mm/addmm/bmm 的参数中提取 M, N, K（和可选的 batch）
    a, b = args[0], args[1]
    config["M"] = a.shape[0]
    config["K"] = a.shape[1]
    config["N"] = b.shape[1]

def _extract_attention_config(args: tuple) -> dict[str, Any]:
    # 从 SDPA 的 query 张量中提取 batch, num_heads, seq_len, head_dim
    q = args[0]
    config["batch"] = q.shape[0]
    config["num_heads"] = q.shape[1]
    config["seq_len"] = q.shape[2]
    config["head_dim"] = q.shape[3]
```

### 核心 Dispatch 模式

```python
class _OperatorGraphTracerMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        func_packet = func._overloadpacket
        packet_name = str(func_packet)

        # 1. 元数据算子: 直接透传
        if packet_name in _METADATA_PACKET_NAMES:
            return func(*args, **kwargs)

        # 2. 视图算子: 执行但不创建节点，追踪别名
        if func_packet in _VIEW_OPS:
            out = func(*args, **kwargs)
            self._storage.register_alias(out, source)
            return out

        # 3. 创建算子: 零耗时节点，注册输出
        if func_packet in _CREATE_OPS:
            out = func(*args, **kwargs)
            # 创建零耗时节点...
            return out

        # 4. 执行算子（跳过 NCCL 集合通信）
        if "c10d" in packet_name:
            out = args[0]  # 原地返回
        else:
            out = func(*args, **kwargs)

        # 5. 分类算子
        op_type, config = _classify_op(func_packet, func, args, kwargs)

        # 6. 收集数据依赖
        for a in flat_args:
            if isinstance(a, torch.Tensor):
                producer = self._storage.get_producer(a)
                if producer is not None:
                    data_deps.append(producer)

        # 7. 流依赖（同流顺序约束）
        prev_op = self._last_op_on_stream.get(stream_id)
        if prev_op is not None:
            stream_deps.append(prev_op)

        # 8. 估算执行时间
        estimated_time_ms = estimate_runtime(...)

        # 9. 创建节点并添加到图
        node = OperatorNode(name=name, op_type=op_type, ...)
        self._graph.add_operator(node)
```

`__torch_dispatch__` 是整个追踪系统的心脏。每当 PyTorch 执行任何算子时，该方法都会被调用。处理逻辑按优先级分为九步：

1. **元数据算子**（`device`、`dtype`、`size` 等）：直接透传，不创建节点。
2. **视图算子**（`reshape`、`transpose` 等）：执行但不创建节点，仅追踪存储别名。
3. **创建算子**（`zeros`、`ones`、`empty` 等）：创建零耗时节点，注册输出存储。
4. **执行算子**：对 NCCL 集合通信返回原始张量（FakeTensor 不能调用真实 NCCL），其他算子正常执行。
5. **分类**：确定算子类型和提取配置。
6. **数据依赖**：通过存储追踪器查找每个输入张量的生产者。
7. **流依赖**：添加同一 CUDA 流上的前序算子依赖。
8. **耗时估算**：调用 compute 子包的 roofline 模型估算执行时间。
9. **建图**：创建 `OperatorNode` 并添加到 `OperatorGraph`。

### 分布式集合通信 No-Op 上下文

```python
@contextlib.contextmanager
def _dist_noop_context():
    import torch.distributed as dist
    orig = {name: getattr(dist, name) for name in (
        "all_reduce", "broadcast", "all_gather",
        "all_gather_into_tensor", "reduce_scatter",
        "reduce_scatter_tensor", "barrier",
    ) if hasattr(dist, name)}

    _handle = _MockDistHandle()
    def _noop(*args, **kwargs):
        return _handle

    try:
        for name in orig:
            setattr(dist, name, _noop)
        yield
    finally:
        for name, fn in orig.items():
            setattr(dist, name, fn)
```

分布式训练框架（如 Megatron-LM）在模型执行中会调用 `torch.distributed` 的集合通信函数。这些函数底层通过 C++ NCCL 绑定实现，无法接受 FakeTensor。因此，此上下文管理器在追踪期间将这些函数替换为 no-op，并返回 `_MockDistHandle` 以支持异步操作的 `.wait()` 调用。

### OperatorGraphTracer 公共接口

```python
class OperatorGraphTracer:
    def trace(self, model, example_inputs, forward_backward=False, loss_fn=None) -> OperatorGraph:
        graph = OperatorGraph(name=type(model).__name__)
        storage_tracker = TensorStorageTracker()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

        # Phase 1: 将模型和输入转为 FakeTensor（无 dispatch mode）
        restore_log = _convert_model_to_fake(model, fake_mode, trace_device)
        fake_inputs = _make_fake_inputs(example_inputs, fake_mode, trace_device)

        try:
            # Phase 2: 激活 dispatch mode 进行追踪
            with _dist_noop_context(), fake_mode, mod_tracker, tracer_mode:
                out = model(*fake_inputs)
                if forward_backward:
                    loss = loss_fn(out)
                    loss.backward()
        finally:
            _restore_model(restore_log)  # 恢复原始模型参数

        return graph
```

追踪过程分为两个阶段：
1. **Phase 1**（无 dispatch mode）：将模型参数和输入转换为 FakeTensor。必须在 dispatch mode 之外完成，避免双重包装。
2. **Phase 2**（激活 dispatch mode）：运行模型前向传播（可选反向传播），所有算子调用被 `_OperatorGraphTracerMode` 拦截并记录到图中。

`finally` 块确保无论追踪是否成功，模型参数都会被恢复。

## 核心类/函数表

| 名称 | 类型 | 用途 |
|------|------|------|
| `_to_fake_device` | function | 将真实张量转为指定设备上的 FakeTensor |
| `_convert_model_to_fake` | function | 将模型所有参数和缓冲区转为 FakeTensor |
| `_restore_model` | function | 恢复模型的原始参数和缓冲区 |
| `_make_fake_inputs` | function | 将用户输入转为 FakeTensor |
| `_storage_key` | function | 获取张量底层存储的唯一标识 |
| `TensorStorageTracker` | class | 追踪张量存储到生产者算子的映射关系 |
| `CUDAEventTracker` | class | 通过 monkey-patch 拦截 CUDA Event 的 record/wait |
| `_classify_op` | function | 将 dispatch 层算子分类为 OperatorType |
| `_extract_gemm_config` | function | 从矩阵乘法参数中提取 M/N/K |
| `_extract_attention_config` | function | 从注意力算子参数中提取形状信息 |
| `_OperatorGraphTracerMode` | class | TorchDispatchMode 实现，核心拦截逻辑 |
| `_MockDistHandle` | class | 模拟异步分布式操作的返回句柄 |
| `_dist_noop_context` | context manager | 将 torch.distributed 集合通信替换为 no-op |
| `OperatorGraphTracer` | class | 面向用户的追踪器，编排整个追踪流程 |

## 与其他模块的关系

```
tracer.py
  |
  +---> config.py             读取 ExecutionMode
  +---> operator_graph.py     构建 OperatorType, OperatorNode, OperatorGraph
  +---> compute/              调用 estimate_runtime 估算算子耗时
  |     +---> compute_cost_predictor.py  获取 _VIEW_OPS, _GEMM_OPS 等算子集合
  |     +---> efficiency_models.py       可选的学习型效率模型
  +<--- api.py                被 api.py 调用
```

- `tracer.py` 是 `operator_graph.py` 的主要消费者（创建节点和图）和 `config.py` 的消费者（读取执行模式）。
- 它还依赖 `compute` 子包进行耗时估算，这部分逻辑通过延迟导入（函数内 import）加载，保持了模块间的松耦合。
- `api.py` 是它唯一的调用者，对外隐藏了追踪引擎的复杂性。

## 小结

`tracer.py` 是 SysSim 的技术核心，巧妙地利用了 PyTorch 内部机制（FakeTensorMode + TorchDispatchMode）实现了零计算开销的模型追踪。其设计中有几个特别值得关注的工程决策：

1. **Storage 级别的依赖追踪**：通过追踪 `untyped_storage` 而非 Tensor 对象，正确处理了 PyTorch 大量使用的视图操作。
2. **分布式 no-op 上下文**：优雅地解决了 FakeTensor 无法通过 NCCL C++ 绑定的问题，使得追踪器可以直接用于 Megatron-LM 等分布式训练框架。
3. **模型参数的保存与恢复**：确保追踪是非侵入式的，不会修改用户的模型状态。
4. **分层的算子过滤**：元数据算子 > 视图算子 > 创建算子 > 计算算子，确保每类算子得到最合适的处理。
