## KV cache

项目里的 KV cache 采用 vLLM 风格的 **PagedAttention + 前缀共享** 实现，整体上分成四层：显存池、逻辑块管理、写入 kernel、读取 kernel。

## 1. 显存池：一次性分配、按层切开

`ModelRunner.__init__` 先 `warmup_model()` 跑一次最大 batch 的 prefill，用 `peak - current` 估算运行时激活峰值，再据此算出能放多少块：

```244:252:myvllm/engine/model_runner.py
        block_bytes = (
            self.block_size
            * 2
            * num_layers
            * num_kv_heads
            * head_dim
            * self.default_dtype.itemsize
        )
        num_available_kv_blocks = int(available_mem // block_bytes)
```

注意 `* 2` 是 K 和 V 共用计数，`num_kv_heads` 已经除以了 `world_size`（TP 时每个 rank 只存自己那几颗 KV head）。

多卡时各 rank 显存不同，用 `all_reduce(MIN)` 取最保守的块数，避免 rank0 的调度器分配超出某个 rank 物理上限的块：

```264:277:myvllm/engine/model_runner.py
        if self.world_size > 1:
            ...
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config["max_cached_blocks"] = per_rank_max_blocks_tensor.item()
```

然后分配一整块连续显存，形状为 `(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)`，再按 layer 切成视图挂到每个 `Attention` 模块上——不复制数据，只是同一块内存的不同偏移：

```286:303:myvllm/engine/model_runner.py
        allocated_kv_cache = torch.zeros(
            2,
            self.config["num_layers"],
            self.config["max_cached_blocks"],
            self.block_size,
            num_kv_heads,
            head_dim,
            device=f"cuda:{self.rank}",
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                layer_id += 1
```

之所以用 `zeros` 而不是 `empty`，是为了避免未写过的 slot 里有 NaN 垃圾值被 attention 读到（prefix cache 未命中的尾部位置仍会被 mask，但脏数据在某些 kernel 路径上会出问题）。

## 2. 逻辑块管理：BlockManager

`Block` 只有四个字段：`block_id / hash / ref_count / token_ids`。三类队列：`free_block_ids`（deque）、`used_block_ids`（set）、`hash_to_block_id`（前缀缓存哈希表）。

```8:36:myvllm/engine/block_manager.py
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.hash = -1
        self.ref_count = 0
        self.token_ids = []
```

**前缀缓存哈希是链式的**：块 hash = `xxhash(前一块 hash || 本块 token_ids)`，只有满块才计算，不满的块 hash 固定为 `-1`（不参与复用）：

```40:45:myvllm/engine/block_manager.py
    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, "little"))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()
```

`allocate()` 逐块查表，命中就 `ref_count += 1` 共享物理块并累加 `seq.num_cached_tokens`，未命中就从 free 队列取新块。命中后还要用 `token_ids` 全量比对一次，防止哈希碰撞误用：

```79:101:myvllm/engine/block_manager.py
            block_id = self.hash_to_block_id.get(h, -1)

            # if cache miss or hash collision
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                no_cache_found = True

            if not no_cache_found:
                seq.num_cached_tokens += self.block_size
                if block_id not in self.used_block_ids:
                    block = self._allocate_block(block_id)
                else:
                    block = self.blocks[self.hash_to_block_id[h]]
                    block.ref_count += 1
```

`deallocate()` 走引用计数，归零才真正回收到 free 队列，所以多个请求共享的相同 prompt 前缀不会被提前释放。注意块被回收时**不清 hash 表**，这就是前缀缓存（跨请求命中）能生效的原因；`_deallocate_block` 只清 `token_ids` 用于下次碰撞校验。

`append()` 处理解码增长：块刚好写满时固化 hash；下一 token 需要新块时（`num_tokens % block_size == 1`）才真正分配。

**抢占**：显存不足时 `Scheduler.preempt` 直接 `deallocate` 整个序列并丢回 waiting 队列，因为 `deallocate` 把 `num_cached_tokens` 归零，该请求下次会按 **recompute** 方式整体重算（没有 swap 到 CPU 的实现）：

```80:83:myvllm/engine/scheduler.py
    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)
```

## 3. 地址映射的关键：`slot_mapping` 和 `block_table`

- `slot_mapping`：**写**用。全局逻辑 slot 编号 `= block_id * block_size + 块内偏移`，把新算出的 K/V 写到哪。
- `block_tables`：**读**用。`(batch, max_num_blocks)` 的二维表，第 `i` 行的第 `j` 项是该序列第 `j` 个逻辑块对应的物理块号，解引用后加上块内偏移即可定位。
- `context_lens`：读的时候的序列长度边界。

`prepare_decode` 里一条序列只处理一个新 token，所以 slot 就是最后一块的最后一个位置：

```401:406:myvllm/engine/model_runner.py
        for seq in seqs:
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))
            slot_mappings.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
```

`prepare_prefill` 复杂一些：只给 `num_cached_blocks` 之后的块生成 slot，且最后一块按 `last_block_num_tokens` 截断（不满块不写满）；如果存在前缀复用（`cu_seqlens_q[-1] < cu_seqlens_k[-1]`）才需要 pad 并下发 `block_tables`：

```339:358:myvllm/engine/model_runner.py
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks :]):
                    if seq.num_cached_blocks + i != seq.num_blocks - 1:
                        slot_mappings.extend(
                            list(
                                range(
                                    block_id * self.block_size,
                                    (block_id + 1) * self.block_size,
                                )
                            )
                        )
                    else:
                        slot_mappings.extend(
                            list(
                                range(
                                    block_id * self.block_size,
                                    block_id * self.block_size
                                    + seq.last_block_num_tokens,
                                )
                            )
                        )
```

这些张量通过全局单例 `Context` 传给各层，避免在 forward 里层层透传参数：

```5:14:myvllm/utils/context.py
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
```

## 4. 写入：Triton scatter kernel

`Attention.forward` 每次先无条件把当前 step 的 K/V 落盘（`slot_mapping` 为 `-1` 的会被 kernel 跳过），再做 attention：

```546:569:myvllm/layers/attention.py
        if (
            k_cache.numel() > 0
            and v_cache.numel() > 0
            and context.slot_mapping is not None
        ):
            ...
            store_kvcache(
                k_to_store,
                v_to_store,
                k_cache,
                v_cache,
                context.slot_mapping,
                self.block_size,
            )
```

kernel 的 grid 是 `(num_tokens, num_kv_heads)`，每个 program 搬一个 token 的一个 head：

```27:35:myvllm/layers/attention.py
    token_idx = tl.program_id(0)  # each GPU thread processes 1 token
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx == -1:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
```

地址计算就是把 `slot` 拆成 `(block_idx, block_offset)` 后在 4 维缓存上做 stride 乘加。缓存布局 `(num_blocks, block_size, num_kv_heads, head_dim)`：

```56:61:myvllm/layers/attention.py
    cache_offset = (
        block_idx * block_size * num_kv_heads * head_dim  # skip previous blocks
        + block_offset * num_kv_heads * head_dim  # skip previous positions in block
        + head_idx * head_dim  # skip previous heads
        + head_offsets
    )
```

这个布局的 trade-off：`(block_size, head, dim)` 排列让**同一 token 的整条 head 向量连续**，store 是完美合并写；但 decode 时同一 head 跨 token 读取的 stride 是 `num_kv_heads * head_dim`，非连续。这也是为什么 decode kernel 退化成逐 token 标量内积（见下）。

## 5. 读取：prefill 用 FlashAttention，decode 用 PagedAttention

**prefill** 时本步的 K/V 就在寄存器里，没必要从缓存再读一遍，所以走独立的 varlen flash attention，`cu_seqlens` 划段、online softmax、GQA 用 `off_h // (num_heads // num_kv_heads)` 映射 KV head：

```142:147:myvllm/layers/attention.py
    # Determine which KV head to use (for GQA)
    kv_head_idx = off_h // (num_heads // num_kv_heads)

    # Load sequence boundaries
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
```

但注意：prefix cache 命中的前缀 token 本步并不参与计算（`input_ids` 只截取 `token_ids[num_cached_tokens:]`），因此 prefill kernel 实际只看得到增量部分。当前实现在这种情况下是**不完整**的——`cu_seqlens_k` 被下发但 prefill kernel 只用了 `cu_seqlens_q`，没有把它们和缓存前缀拼接起来。这是这个 nano 实现最明显的简化点。

**decode** 走 `paged_attention_decode_kernel`，两层循环：外层按 `BLOCK_N` 分块遍历 context，内层逐 token 用 `block_table` 解引用物理块号：

```377:401:myvllm/layers/attention.py
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size

                    if block_num < max_num_blocks:
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(
                            block_tables_ptr + block_table_offset
                        )

                        if physical_block_idx != -1:
                            k_offset = (
                                physical_block_idx
                                * block_size
                                * num_kv_heads
                                * head_dim
                                + block_offset * num_kv_heads * head_dim
                                + kv_head_idx * head_dim
                                + offs_d
                            )
                            k_vec = tl.load(k_cache_ptr + k_offset)
```

然后沿用标准的 online softmax 三段式（`m_i / l_i / acc`），最后 `output = acc / l_i`。这里为了可读性牺牲了性能：内层是串行 for + `tl.sum(q * k_vec)` 标量乘，不是 `tl.dot` 的矩阵块乘，所以 decode 吞吐远低于生产级 kernel。

用一张图概括一次 decode step 的数据流：

```
seq.block_table = [3, 7, 12]        block_size = 4
                    │  │   └─ 逻辑块 2 → 物理块 12
                    │  └───── 逻辑块 1 → 物理块 7
                    └──────── 逻辑块 0 → 物理块 3

slot_mapping  = [12*4 + 1]  → 新 token 写到物理块 12 的第 1 个槽
block_tables  = [[3, 7, 12, -1, ...]]   → kernel 按此读取历史 K/V
context_lens  = [10]         → 只读前 10 个 token
```

## 6. 与 CUDA Graph 的配合

decode 时 KV 相关的三个输入张量必须在 capture 前预分配好固定地址，否则 graph 无法复用。所以 `capture_cudagraph` 按 `max_bs × max_num_blocks` 预分配 `slot_mapping / context_lens / block_tables`，然后对 `[1,2,4,8] + range(16, max_bs+1, 16)` 每个 batch size 各捕获一份：

```507:515:myvllm/engine/model_runner.py
        # for paged attention
        # where to write new KV values in the cache
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f"cuda:{self.rank}")
        # how many tokens each sequence has processed
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f"cuda:{self.rank}")
        # where to read KV values in the cache
        block_tables = torch.zeros(
            max_bs, max_num_blocks, dtype=torch.int32, device=f"cuda:{self.rank}"
        )
```

运行时选最小的够用 graph，用 copy_ 灌数据再 `replay()`；`slot_mapping` 先 `fill_(-1)` 是为了让 padding 的 batch 位置不写缓存（配合 kernel 里的 `if slot_idx == -1: return`）：

```459:471:myvllm/engine/model_runner.py
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars
            # copy input data into graph variables
            vars["input_ids"][:bs].copy_(input_ids)
            vars["slot_mapping"][:bs].fill_(-1)
            vars["slot_mapping"][:bs].copy_(context.slot_mapping)
            vars["context_lens"].zero_()
            vars["context_lens"][:bs].copy_(context.context_lens)
            vars["block_tables"][:bs, : context.block_tables.size(1)] = (
                context.block_tables
            )
            graph.replay()
```

## 7. 设计要点小结

| 关注点 | 本实现做法 |
|---|---|
| 碎片 | 逻辑块 → 物理块二级映射，块内可不满 |
| 共享 | `ref_count` 引用计数 + 链式 xxhash 前缀缓存 |
| 跨请求复用 | 块回收时保留 `hash_to_block_id` 项，只清 `token_ids`（同时用于碰撞校验） |
| 换出 | 未实现 swap，`preempt` 直接丢弃并 recompute |
| 扩展 | 块满时才固化 hash，未满块 hash=-1 不参与共享 |
| TP | `num_kv_heads // world_size`，各 rank 独立分配并 MIN-all_reduce 对齐 |

需要继续深挖的话，有两处值得注意的薄弱点：一是 prefill 阶段没有把 prefix cache 中的历史 K/V 拼进 attention（只看增量），二是 `paged_attention_decode_kernel` 的逐 token 内积写法，换成 block 级 `tl.dot` 会有数量级的性能提升。

---

继续，下面按“收益从大到小”给出具体改法。先定位当前 kernel 慢在哪：

| 问题 | 位置 | 影响 |
|---|---|---|
| 内层逐 token 标量循环 | `for i in range(BLOCK_N)` 两次 | 指令数 = BLOCK_N × (block_table load + 1 个 256B load + 归约)，编译器无法向量化，无法用 `num_stages` 流水 |
| 循环上界用 `max_num_blocks * block_size` | `max_chunks` | 短序列要空转整个 batch 最长上下文 |
| 每个 query head 都重读 KV | grid = `(B, num_heads)` | GQA 下 K/V 被读 `num_heads/num_kv_heads` = 4 次 |
| 无 split-K | grid 只有 `B × H` | batch=1 时只有 32 个 program，无法打满 HBM 带宽 |

## 改进 1（必做）：按块 tile 化，一次搬一整块

核心思路：**以 `BLOCK_N` 个 token 为一个 tile，一次 `tl.load` 出 `(BLOCK_N, head_dim)` 的 K/V，用一次向量归约代替 BLOCK_N 次标量内积**。因为 `BLOCK_N` 整除 `block_size`，一个 tile 必落在同一个物理块内，`block_table` 只需查一次。

```python
@triton.jit
def paged_attention_decode_kernel(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr,
    block_tables_ptr, context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,          # 必须整除 block_size
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    context_len = tl.load(context_lens_ptr + batch_idx)
    # 关键：循环上界跟本序列长度走，不再用 max_num_blocks
    num_chunks = tl.cdiv(context_len, BLOCK_N)

    offs_d = tl.arange(0, head_dim)
    offs_n = tl.arange(0, BLOCK_N)

    q = tl.load(
        q_ptr + batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    )

    m_i = -1e30
    l_i = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    token_stride = num_kv_heads * head_dim           # 旧布局 (block, bs, kvh, d)
    head_base = kv_head_idx * head_dim

    for c in range(num_chunks):
        n0 = c * BLOCK_N
        logical_blk = n0 // block_size               # 整个 tile 属于同一个逻辑块
        in_block = n0 % block_size
        phys = tl.load(block_tables_ptr + batch_idx * max_num_blocks + logical_blk)

        offs_tok = n0 + offs_n
        mask_n = offs_tok < context_len

        # (BLOCK_N, head_dim) 二维偏移：一次完成整块定位
        kv_off = (
            phys * block_size * token_stride
            + (in_block + offs_n[:, None]) * token_stride
            + head_base
            + offs_d[None, :]
        )
        k = tl.load(k_cache_ptr + kv_off, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_cache_ptr + kv_off, mask=mask_n[:, None], other=0.0)

        # 一次归约代替 BLOCK_N 次 tl.sum(q * k_vec)
        qk = tl.sum(k * q[None, :], axis=1) * scale
        qk = tl.where(mask_n, qk, -1e30)

        m_new = tl.maximum(m_i, tl.max(qk))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p)
        m_i = m_new

    tl.store(
        out_ptr + batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d,
        acc / l_i,
    )
```

launcher 里把调参放开：

```python
BLOCK_N = 32 if head_dim >= 128 else 64     # fp32 累加器压力
grid = (batch_size, num_heads)
paged_attention_decode_kernel[grid](
    ...,
    BLOCK_N=tl.constexpr(BLOCK_N),
    num_warps=4,
    num_stages=2,        # 让 K/V 的 load 与计算重叠，这是标量循环做不到的
)
```

注意 `BLOCK_N` 不要盲目取大：`tl.sum(p[:, None] * v, axis=0)` 会物化 `(BLOCK_N, head_dim)` 的 fp32 中间量，`BLOCK_N=64, head_dim=128` 就是 32KB 寄存器/线程块，容易溢出。从 32 开始试。

## 改进 2：GQA 组内共享 K/V（省 4 倍带宽）

decode 是**带宽瓶颈**，而当前每个 query head 都独立读一遍同一份 K/V。把 grid 改成 `(B, num_kv_heads)`，一个 program 负责整组 `G = num_heads // num_kv_heads` 个 query head：K/V tile 只加载一次，G 个头复用。因为 `G` 是 `tl.constexpr`，用 Python 层循环展开即可：

```python
    group = num_heads // num_kv_heads          # constexpr，Python 层展开
    offs_g = tl.arange(0, group)

    # 一次载入整组 query: (group, head_dim)
    q = tl.load(
        q_ptr + batch_idx * num_heads * head_dim
        + (kv_head_idx * group + offs_g)[:, None] * head_dim
        + offs_d[None, :]
    )

    m_i = tl.full([group], -1e30, tl.float32)
    l_i = tl.zeros([group], tl.float32)
    acc = tl.zeros([group, head_dim], tl.float32)

    for c in range(num_chunks):
        ...  # k, v 只加载一次
        # (group, BLOCK_N)：每行一个 query head 对该 tile 的分数
        qk = tl.sum(k[None, :, :] * q[:, None, :], axis=2) * scale
        qk = tl.where(mask_n[None, :], qk, -1e30)

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])                    # (group, BLOCK_N)

        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
```

`(group, BLOCK_N)` 的中间量比 `(group, BLOCK_N, head_dim)` 小得多，所以 `BLOCK_N` 反而要调小到 16~32。

## 改进 3：split-K（flash decoding），解决 batch 小打不满 GPU

batch=1、上下文 4K 时，grid 只有 `1 × 32` 个 program，SM 空转，而每个 program 要串行读完 4K 个 token 的 K/V。做法是把 KV 维切段：

```python
NUM_SPLITS = 8
grid = (batch_size, num_heads, NUM_SPLITS)
```

每个 split 只处理 `[s*chunk, (s+1)*chunk)` 的 token，输出自己的部分结果 `(m, l, acc)`，再用第二个 kernel 做 log-sum-exp 合并：

```
m_final = max_s m_s
acc_final = Σ_s acc_s * exp(m_s - m_final) / l_final
l_final   = Σ_s l_s * exp(m_s - m_final)
```

scratch 形状 `(num_splits, B, H, D+2)`，第二次归约开销可忽略。这条改动在**低并发、长上下文**场景是数量级提升，高并发时收益变小甚至略亏（多一次 kernel），建议按 `context_len` 和当前 batch 动态选 `NUM_SPLITS`。

## 改进 4（可选）：换缓存布局，让 tile 完全连续

当前 `(num_blocks, block_size, num_kv_heads, head_dim)` 下，一个 tile 的行 stride 是 `num_kv_heads * head_dim`（256B×4=2KB 跳一步），tile 不是连续内存。改成 `(num_blocks, num_kv_heads, block_size, head_dim)` 后，同一 head 的一个块是 `block_size × head_dim` 的**完全连续**区间，可以单次大事务读取，L2 预取也更友好。需要同步改三处：

```python
# 1) attention.py: store_kvcache_kernel 的 cache_offset
    cache_offset = (
        block_idx * num_kv_heads * block_size * head_dim
        + head_idx * block_size * head_dim
        + block_offset * head_dim
        + head_offsets
    )
```

```python
# 2) attention.py: decode kernel 的 kv_off
    kv_off = (
        phys * num_kv_heads * block_size * head_dim
        + kv_head_idx * block_size * head_dim
        + (in_block + offs_n[:, None]) * head_dim
        + offs_d[None, :]
    )
```

```python
# 3) model_runner.py: allocate_kv_cache 的形状（只是视图，但要对齐文档/校验）
        allocated_kv_cache = torch.zeros(
            2,
            self.config["num_layers"],
            self.config["max_cached_blocks"],
            num_kv_heads,          # 提到 block_size 之前
            self.block_size,
            head_dim,
            ...
        )
```

代价：store 侧（prefill 批量写）从“每 token 连续写整条 head”变成“每 head 写 `block_size × head_dim`”，两者其实都是连续的，只是写入序不同，实测差别不大。

## 其他细节

- `context_lens` 用 `int32` 而不是 `int64`，减少比较/索引的 ALU 开销（`prepare_decode` 里 dtype 改成 `torch.int32` 即可）。
- 调度器侧按 `context_len` 排序 batch（vLLM 的 `sort_by_length`），让相近长度的序列同批，减少 tile 空转。
- `tl.dot` 在这个场景**不是必需的**：`M=1` 时 triton 要求 `M≥16`，硬凑 16 行有 15 行浪费，而 kernel 本身是带宽瓶颈、不是算力瓶颈。除非你把 16 个 query head 拼到 M 维（`num_heads=32` 时刚好两组），才值得上 tensor core。
- `block_size` 取值：当前默认 256 对这个 kernel 偏大（tile 太粗），要么靠 `BLOCK_N` 内置子切分（改进 1 已支持），要么直接调到 16/32。

## 验证方法

`benchmark_decoding.py` 里已经有可信的 golden 实现，直接拿来做正确性对拍，不要凭手感判断：

```python
ref = decode_torch_optimized(q, k_cache, v_cache, block_tables,
                             context_lens, scale, num_heads, num_kv_heads,
                             head_dim, block_size)
out = paged_attention_decode_triton(...)
print((ref - out).abs().max())   # fp16 下应 < 1e-2
```

建议的落地顺序：**先只做改进 1**（改动局限在单个 kernel，风险最低，通常就能拿到 3–5x），对拍通过后再叠加改进 2，最后按需上 split-K。要不要我直接把改进 1 的版本写进 `myvllm/layers/attention.py`，并同时把 `benchmark_decoding.py` 里的对拍 benchmark 更新成对比新旧两版？
