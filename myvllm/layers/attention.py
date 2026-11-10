"""
Attention + KV Cache 写入 + FlashAttention 分派

【第一性原理】Transformer 的 attention：Q,K,V = 当前步的 q,k,v；
对 prefill 是整段 prompt 的 q,k,v，
对 decode 是「当前 1 个 token 的 q,k,v」加上「历史 k,v 从 cache 里读」。
为不重复算历史，每步把新算出的 k,v 按 slot_mapping 写进 k_cache/v_cache（PagedAttention：物理块 + block_table 逻辑块映射）。

【通俗理解】Attention 就是「用当前 token 的 Q 去问历史的 K，按相似度加权历史的 V 得到输出」。
K/V 要存下来（KV cache），否则 decode 时每步都要重算整段历史。
本层先把手里的 k,v 按 slot 写进 cache，再根据是 prefill 还是 decode 调用不同 kernel：
prefill 用变长多序列的接口，decode 用「Q + 从 cache 读 K/V」的接口。

本模块职责：
1. 用 Triton kernel store_kvcache_kernel 把本步的 key/value 按 slot_mapping 写入 k_cache/v_cache。
2. 根据 context.is_prefill 与 block_tables 选择：
   - Prefill + 无 prefix cache：flash_attn_varlen_func(q,k,v, cu_seqlens_q/k, ...)
   - Prefill + prefix cache：从 cache 里读已有 k,v，再和本步新 q 一起做 varlen attention，block_table 指定物理块。
   - Decode：flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, block_table, ...)
详见教案 06、07、13c。slot_mapping 与 PagedAttention：每个逻辑 token 对应一个 slot = 在全局 KV cache 中的线性下标，由 prepare_* 里 block_table 与块内偏移算出，与 block_table 的「逻辑块→物理块」映射一致。
"""

import torch
import triton
import triton.language as tl
from torch import nn

from myvllm.utils.context import get_context


@triton.jit  # 编译为 GPU kernel；调用时 store_kvcache_kernel[(N,)](...) 表示 1D grid，program_id(0) 从 0 到 N-1
def store_kvcache_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    """
    每个 program 处理一个「逻辑 token」的 key/value 向量。
    根据 slot_mapping_ptr[idx] 得到该 token 在物理 cache 中的 slot，
    把 key/value 写入 k_cache/v_cache。
    slot == -1 表示不写（占位或 padding）。
    D = num_heads * head_dim，即单 token 的 k/v 展平长度。
    Grid 布局：(num_tokens, num_kv_heads)\n
    Cache 布局：(num_blocks, block_size, num_kv_heads, head_dim)
    """
    # 每个 program 搬一个 token 的一个 head：
    # thread ID, in dimension 0
    token_idx = tl.program_id(0)  # 当前 GPU 线程负责第 idx 个 token 的 K/V 写入
    slot_idx = tl.load(
        slot_mapping_ptr + token_idx
    )  # 该 token 在物理 cache 中的线性 slot（-1 表示不写）
    if slot_idx == -1:
        return

    # Calculate which block and position within block
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # Process each head
    # program_id(0) = which token
    # program_id(1) = which head
    head_idx = tl.program_id(1)

    # it creates a vector [0, 1, ..., head_dim-1]
    # Load key and value for this token and head
    head_offsets = tl.arange(0, head_dim)
    # Input: (num_tokens, num_kv_heads, head_dim)
    # example: input_offset = 5 * (8 * 128) + 3 * 128 + [0, 1, 2, ..., 127]
    #         = 5120 + 384 + [0, 1, 2, ..., 127]
    #         = [5504, 5505, 5506, ..., 5631]
    input_offset = (
        token_idx * num_kv_heads * head_dim  # skip previous tokens
        + head_idx * head_dim  # skip previous heads
        + head_offsets
    )

    # 地址计算就是把 slot 拆成 (block_idx, block_offset) 后在 4 维缓存上做 stride 乘加。
    # 缓存布局 (num_blocks, block_size, num_kv_heads, head_dim)
    # 这个布局的 trade-off：(block_size, head, dim) 排列让同一 token 的整条 head 向量连续，store 是完美合并写；
    # 但 decode 时同一 head 跨 token 读取的 stride 是 num_kv_heads * head_dim，非连续。
    # 这也是为什么 decode kernel 退化成逐 token 标量内积。
    cache_offset = (
        block_idx * block_size * num_kv_heads * head_dim  # skip previous blocks
        + block_offset * num_kv_heads * head_dim  # skip previous positions in block
        + head_idx * head_dim  # skip previous heads
        + head_offsets
    )

    # load key and value value floats from the pointers's memory
    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)

    # store into cache
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
):
    """
    把本步算出的 key/value 按 slot_mapping 写入 paged k_cache/v_cache。

    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) - 每个 token 对应的 cache slot（线性下标）
        block_size: 每个物理块容纳的 token 数

    【为什么可以避免整段拷贝 / .contiguous() 的代价】
    key 来自 qkv → split → view 的链条（见 qwen3.py Qwen3Attention.forward），
    它通常不是"全新分配"的连续张量，而是嵌在 qkv 大张量里的一段 view，
    其 stride 一般形如 (qkv_row_size, head_dim, 1)：stride[0] 很大（跨过 Q 部分），
    但 stride[1] == head_dim、stride[2] == 1，即「head 内连续、head 间步长正确」。
    kernel 的地址公式假设输入按 (token, head, dim) 连续排布，
    所以下面先对不满足假设的 key/value 调用 .contiguous()（大多数情况已满足，零拷贝）；
    只有当 view 的 stride 与 kernel 假设冲突时才会触发一次真实拷贝。
    """

    num_tokens, num_kv_heads, head_dim = key.shape

    # kernel 按「token 内 head 连续、head 内元素连续」的布局计算地址，
    # 若输入 view 不满足该假设（如极少数非标准排布），退化为一次显式拷贝
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()

    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert slot_mapping.numel() == num_tokens, (
        "Slot mapping size must match number of tokens"
    )

    grid = (num_tokens, num_kv_heads)

    # launch num_tokens x num_kv_heads threads
    store_kvcache_kernel[grid](
        key,  # tensors are automatically converted to pointers by triton
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads=tl.constexpr(num_kv_heads),
        head_dim=tl.constexpr(head_dim),
        block_size=tl.constexpr(block_size),
    )


@triton.jit
def flash_attention_varlen_kernel(
    Q,
    K,
    V,
    O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    用于变长序列的 Flash Attention 内核。
    每个程序处理一个序列中一个注意力头的一个查询块。
    """
    # 启动维度划分：
    # axis 0 = 查询块编号（每 BLOCK_M 个 query 一个 program）
    # axis 1 = 注意力头编号（GQA 下按 Q head 计）
    # axis 2 = batch 里的序列编号
    start_m: int = tl.program_id(0)  # 当前处理的 query 块索引
    off_h: int = tl.program_id(1)  # 当前处理的 head 索引
    seq_idx: int = tl.program_id(2)  # 当前处理的序列索引（batch 内第几条 prompt）

    # GQA 映射：num_heads 个 Q head 共享 num_kv_heads 个 KV head，
    # 每 group = num_heads // num_kv_heads 个 Q head 复用同一个 KV head
    kv_head_idx = off_h // (num_heads // num_kv_heads)

    # 载入本序列在 varlen 打包中的起止边界：
    # cu_seqlens 是 (num_seqs+1,) 的前缀和，seq_start/seq_end 即本序列
    # 在打平 token 维度上的 [start, end) 区间
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start

    # 当前 query 块已超出序列长度，这个 program 无事可做，提前退出
    if start_m * BLOCK_M >= seq_len:
        return

    # 本块 query 的偏移：offs_m 为 BLOCK_M 行的行下标，offs_d 为 head 内维度下标
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    # Q 指针：Q 形状 (total_tokens, num_heads, head_dim)
    q_ptrs = (
        Q
        + (seq_start + offs_m[:, None]) * num_heads * head_dim
        + off_h * head_dim
        + offs_d[None, :]
    )

    # 载入 Q 块 - 形状 (BLOCK_M, head_dim)；越界行用 0 填充
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # ---- online softmax 状态初始化 ----
    # l_i: softmax 分母的累积和；m_i: 每行已见的最大 logit（初始化为极小值）；
    # acc: 加权 V 的未归一化累积和。三者构成 FlashAttention 的 online softmax 状态
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # K/V 需要分 BLOCK_N 大小的块循环处理（序列可能长于一个块）
    num_blocks = tl.cdiv(seq_len, BLOCK_N)

    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # 当前 K 块内有效位置的 mask（尾部块可能不满）
        mask_n = offs_n < seq_len

        # K 指针：K 形状 (total_tokens, num_kv_heads, head_dim)
        # 转置读取 (head_dim, BLOCK_N) 以便和 q 做 tl.dot
        k_ptrs = (
            K
            + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim
            + kv_head_idx * head_dim
            + offs_d[:, None]
        )

        # 载入 K 块 - 形状 (head_dim, BLOCK_N)
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        # 计算 QK^T - 形状 (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)
        qk = qk * scale

        # 因果 mask：query 位置只能 attend 到 <= 自身的位置
        # （两边都加 seq_start 是冗余的，效果等价于 offs_m >= offs_n）
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)

        # ---- online softmax 增量更新 ----
        # m_ij: 当前块内每行最大 logit；m_i_new: 与历史最大值取 max
        # alpha: 历史累积量需要缩放的因子 exp(旧max - 新max)
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        # p: 当前块的未归一化注意力权重 exp(qk - 新max)
        p = tl.exp(qk - m_i_new[:, None])

        # 用 alpha 重新缩放历史累积的 acc
        acc = acc * alpha[:, None]

        # 载入 V 块 - 形状 (BLOCK_N, head_dim)
        v_ptrs = (
            V
            + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim
            + kv_head_idx * head_dim
            + offs_d[None, :]
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # 累加加权值：acc += p @ V
        acc = acc + tl.dot(p.to(v.dtype), v)

        # 更新 softmax 分母与最大值
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # 最终归一化：除以 softmax 分母
    acc = acc / l_i[:, None]

    # 写回输出：O 形状 (total_tokens, num_heads, head_dim)
    o_ptrs = (
        O
        + (seq_start + offs_m[:, None]) * num_heads * head_dim
        + off_h * head_dim
        + offs_d[None, :]
    )
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Prefill 阶段的变长 FlashAttention 封装。

    多条不同长度的 prompt 被打平成 (total_tokens, ...) 一个大张量，
    cu_seqlens（前缀和，形状 (num_seqs+1,)）负责划出每条序列的边界。
    kernel 的 grid = (ceil(max_len/BLOCK_M), num_heads, num_seqs)，
    每个 program 处理「某条序列 × 某个 head × BLOCK_M 个 query」。

    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens: 每条序列在 token 维上的累计长度（前缀和）
        scale: attention 缩放因子（已含 1/sqrt(head_dim)）

    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # kernel 地址公式假设 token 维连续，必要时做一次拷贝
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # 预分配输出
    output = torch.empty_like(q)

    # 根据 head_dim 保守选择块大小，避免 shared memory 超限：
    # shared memory 用量约 BLOCK_M * BLOCK_N * 4（float32 注意力分数）
    # + BLOCK_M * head_dim * 4（Q）
    # + BLOCK_N * head_dim * 4（K, V）
    # 目标是大多数 GPU 上总占用 < 48KB

    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16

    # 序列条数
    num_seqs = cu_seqlens.shape[0] - 1

    # 取最长序列长度以确定 grid 第一维（把 grid 一次性铺满，无需 host 端逐序列循环）
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()

    # 计算 grid 维度：axis0 = 最长序列上的 query 块数, axis1 = head 数, axis2 = 序列数
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)

    flash_attention_varlen_kernel[grid](
        q,
        k,
        v,
        output,
        cu_seqlens,
        scale,
        num_heads=tl.constexpr(num_heads),
        num_kv_heads=tl.constexpr(num_kv_heads),
        head_dim=tl.constexpr(head_dim),
        BLOCK_M=tl.constexpr(BLOCK_M),
        BLOCK_N=tl.constexpr(BLOCK_N),
    )

    return output


@triton.jit
def paged_attention_decode_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,  # KV 分块大小；要求 block_size % BLOCK_N == 0（tile 不能跨逻辑块）
):
    """
    Decode 阶段的 PagedAttention kernel（教学实现）。

    decode 时每条序列只有 1 个新 query，但要与 cache 里的全部历史 K/V 做 attention。
    历史 K/V 散落在物理块里，通过 block_tables（逻辑块号 -> 物理块号）解引用。
    每个 program 处理「batch 里的一条序列 × 一个 head」，
    以 BLOCK_N 为单位分块遍历 KV，用 online softmax 累积结果。
    """
    batch_idx = tl.program_id(0)  # batch 中第几条序列
    head_idx = tl.program_id(1)  # 第几个 Q head

    # GQA 映射：多个 Q head 共享一个 KV head
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    # 载入本序列的历史上下文长度（含当前步刚写入的 token）
    context_len = tl.load(context_lens_ptr + batch_idx)
    # 关键：循环上界跟本序列长度走，不再用 max_num_blocks
    num_chunks = tl.cdiv(context_len, BLOCK_N)

    # 载入本步唯一的一个 query: (batch_size, num_heads, head_dim)
    offs_d = tl.arange(0, head_dim)
    offs_n = tl.arange(0, BLOCK_N)

    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)

    # online softmax 状态（decode 只有 1 个 query，状态退化为标量）
    m_i = -1e10
    l_i = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    # 计算按 cache 最大容量算的总 chunk 数（教学版本用它做循环上界）
    # max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)

    token_stride = (
        num_kv_heads * head_dim
    )  # cache 布局 (block, bs, kvh, d) 中一个 token 的步长
    head_base = kv_head_idx * head_dim

    """ 教学实现（当前未启用，仅用于理解思路，为可读性牺牲了性能）
    # 两层循环：外层按 BLOCK_N 分块遍历 context，
    # 内层逐 token 用 block_table 解引用物理块号，每次只算一个标量内积。
    # 生产版本（下方代码）把内层循环向量化成 (BLOCK_N, head_dim) 的矩阵操作。
    for chunk_idx in range(max_chunks):
        # 本 chunk 覆盖的全局 token 区间起点
        token_start = chunk_idx * BLOCK_N

        # 起点已超出本序列长度则跳过
        if token_start < context_len:
            # 本 chunk 内有效 token 的 mask（尾部可能不满）
            offs_n = token_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len

            # 注意力分数数组，无效位置初始化为 -1e10
            qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10

            # 逐 token 载入 K 并计算分数
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    # 把全局 token 下标拆成 (逻辑块号, 块内偏移)
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size

                    if block_num < max_num_blocks:
                        # 查 block_table 得到物理块号
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(
                            block_tables_ptr + block_table_offset
                        )

                        if physical_block_idx != -1:
                            # 按 (物理块, 块内偏移, head, dim) 计算缓存内偏移，载入 K 向量
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

                            # 该 token 的注意力分数（标量内积）
                            score = tl.sum(q * k_vec) * scale

                            # 用 tl.where 把分数写进 qk 的第 i 个位置
                            mask_i = tl.arange(0, BLOCK_N) == i
                            qk = tl.where(mask_i, score, qk)

            # 无效位置统一置为极小值
            qk = tl.where(mask_n, qk, -1e10)

            # online softmax 增量更新
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)

            # 用 alpha 缩放历史累积量
            acc = acc * alpha
            l_i = l_i * alpha

            # 逐 token 载入 V 并加权累加
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size

                    if block_num < max_num_blocks:
                        # 查 block_table 得到物理块号
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(
                            block_tables_ptr + block_table_offset
                        )

                        if physical_block_idx != -1:
                            # 载入 V 向量
                            v_offset = (
                                physical_block_idx
                                * block_size
                                * num_kv_heads
                                * head_dim
                                + block_offset * num_kv_heads * head_dim
                                + kv_head_idx * head_dim
                                + offs_d
                            )
                            v_vec = tl.load(v_cache_ptr + v_offset)

                            # 从 p 里取出该 token 的权重
                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))

                            acc = acc + weight * v_vec
                            l_i = l_i + weight

            m_i = m_i_new
    """

    # ---- 生产实现：向量化版 ----
    for chunk_idx in range(num_chunks):
        token_start = chunk_idx * BLOCK_N
        logical_blk = token_start // block_size  # 整个 tile 属于同一个逻辑块
        in_block = token_start % block_size
        # 查 block_table：本 tile 的所有 token 共享同一个物理块
        phys = tl.load(block_tables_ptr + batch_idx * max_num_blocks + logical_blk)

        offs_tok = token_start + offs_n
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

        # 一次归约代替教学版的 BLOCK_N 次标量内积：q 与整块 K 逐行做内积
        qk = tl.sum(k * q[None, :], axis=1) * scale
        # 无效位置置为极小值，softmax 后权重趋近 0
        qk = tl.where(mask_n, qk, -1e30)

        # online softmax 增量更新
        m_new = tl.maximum(m_i, tl.max(qk))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)

        # acc += p @ V（一次矩阵-向量加权求和）
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p)
        m_i = m_new

    # 最终归一化：除以 softmax 分母
    output = acc / l_i

    # 写回输出
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


"""
=== Benchmark Results ===
minivllm:
  latency: 4.2743
  tokens: 336.0000
  tps: 78.6094
transformers:
  latency: 13.7076
  tokens: 768.0000
  tps: 56.0273
"""


def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """
    Decode 阶段：基于 paged KV cache 计算 attention 的封装。

    decode 时每条序列只有 1 个 query token，K/V 全部从 cache 里读，
    通过 block_tables 间接寻址散落的物理块。

    Args:
        query: (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (batch_size, max_num_blocks) 每条序列的 逻辑块号 -> 物理块号 映射
        context_lens: (batch_size,) 每条序列的历史上下文长度
        scale: attention 缩放因子（已含 1/sqrt(head_dim)）

    Returns:
        output: (batch_size, num_heads, head_dim)
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]

    # 保证 token/head 维连续，满足 kernel 的地址公式
    query = query.contiguous()

    output = torch.empty_like(query)

    # KV 的分块大小；注意需满足 block_size % BLOCK_N == 0（tile 不能跨逻辑块）
    BLOCK_N = 64 if head_dim <= 128 else 32

    # grid: axis0 = batch 中每条序列, axis1 = 每个 head
    grid = (batch_size, num_heads)

    paged_attention_decode_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=tl.constexpr(scale),
        num_heads=tl.constexpr(num_heads),
        num_kv_heads=tl.constexpr(num_kv_heads),
        head_dim=tl.constexpr(head_dim),
        block_size=tl.constexpr(block_size),
        max_num_blocks=tl.constexpr(max_num_blocks),
        BLOCK_N=tl.constexpr(BLOCK_N),
    )

    return output


class Attention(nn.Module):
    """
    多头注意力：接收本层算好的 q,k,v；先按 context 写 KV cache（若有），
    再根据 prefill/decode 和是否 prefix cache 调用对应的 FlashAttention 接口，
    返回 attention 输出。
    k_cache / v_cache 在 ModelRunner.allocate_kv_cache 里按层绑定到全局 KV cache 张量的对应层切片。
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        block_size: int = 16,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        # 空 tensor 占位。真正的 KV cache 显存由 ModelRunner.allocate_kv_cache 分配:
        # 1. __init__ 阶段: 仅占位 (torch.tensor([]))，不占显存
        # 2. 模型加载 + warmup forward: 触发 cuBLAS/cudNN workspace 分配, 测峰值显存
        # 3. allocate_kv_cache: 根据 (total*util - used - peak + current) / block_bytes 算 num_blocks
        #    分配全局大 tensor (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        #    遍历 model.modules() 找到每个 Attention, 按层切片绑定:
        #      module.k_cache = self.kv_cache[0, layer_id]
        #      module.v_cache = self.kv_cache[1, layer_id]
        # 所有层共享同一个大 tensor, 按层做 view 切片 (不拷贝)。
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # ---- 第 1 步：先把当前 step 的 K/V 写入 cache（若 cache 已分配） ----
        # 无条件先落盘（slot_mapping 为 -1 的 token 会被 kernel 自动跳过），再做 attention
        if (
            k_cache.numel() > 0
            and v_cache.numel() > 0
            and context.slot_mapping is not None
        ):
            # 统一成 kernel 期望的形状: (num_tokens, num_kv_heads, head_dim)
            if k.dim() == 4:
                # 带 batch 维: (B, N, num_kv_heads, head_dim) -> 打平为 (B*N, num_kv_heads, head_dim)
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                # 已经是 (num_tokens, num_kv_heads, head_dim)
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()

            store_kvcache(
                k_to_store,
                v_to_store,
                k_cache,
                v_cache,
                context.slot_mapping,
                self.block_size,
            )

        # 缩放因子 = 1/sqrt(head_dim)（self.scale 默认为 1.0）
        scale = self.scale / (self.head_dim**0.5)

        # ---- 第 2 步：按 prefill / decode 分派不同的 attention kernel ----
        if context.is_prefill:
            # Prefill：变长 FlashAttention，K/V 直接用本步现算的（不必从 cache 读回）
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")

            o = flash_attention_prefill(
                q,
                k,
                v,
                cu_seqlens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
            # 输出: (total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            # Decode：PagedAttention，K/V 从 paged cache 里按 block_table 读取
            assert context.block_tables is not None, "context.block_tables must be set"
            assert context.context_lens is not None, "context.context_lens must be set"
            o = paged_attention_decode(
                q,
                k_cache,
                v_cache,
                context.block_tables,
                context.context_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
            )
            # o: (batch_size, num_heads, head_dim) -> (batch_size, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)


if __name__ == "__main__":
    # 简单的 decode 路径性能自测脚本
    from myvllm.utils import reset_context, set_context

    layer = Attention(num_heads=8, head_dim=64).cuda()
    B, N, D = 4, 1024, 512

    # 模拟 decode 输入：每条序列只有 1 个新 token 的 q/k/v
    q = torch.randn(B, 1, D).cuda()
    k = torch.randn(B, 1, D).cuda()
    v = torch.randn(B, 1, D).cuda()

    # 直接挂一个简单 cache（注意：形状并非真实的 paged cache，仅用于跑通流程）
    layer.k_cache = torch.zeros(B, N, D).cuda()
    layer.v_cache = torch.zeros(B, N, D).cuda()

    # 构造 decode 阶段的 context：4 条序列，每条历史长度 N//4
    cu_seqlens = torch.tensor(
        [0, N // 4, N // 2, 3 * N // 4, N], dtype=torch.int32
    ).cuda()
    context_lens = torch.tensor(
        [N // 4, N // 4, N // 4, N // 4], dtype=torch.long
    ).cuda()
    slot_mapping = context_lens.clone()
    block_tables = torch.zeros(B, (N + 255) // 256, dtype=torch.int32).cuda()

    for _ in range(10):  # 预热（含 Triton JIT 编译）
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        _ = layer(q, k, v)
        reset_context()

    import time

    times = []
    for _ in range(100):  # 计时迭代
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(q, k, v)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
        reset_context()

    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
