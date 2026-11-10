# 模型执行器（ModelRunner）：推理引擎中真正"跑模型"的一层。
#
# 核心职责：
#   1. 初始化分布式环境（NCCL 进程组），并根据模型名构建对应的 CausalLM 模型；
#   2. 加载预训练权重，将模型搬到 GPU 上；
#   3. 预热模型以探测显存峰值，再据此分配 Paged Attention 用的 KV 缓存池；
#   4. （可选）为 decode 阶段捕获 CUDA Graph，消除 kernel 启动开销；
#   5. 为 prefill / decode 两个阶段准备输入张量（input_ids、slot_mapping、
#      block_tables、cu_seqlens 等），执行前向并采样出下一个 token；
#   6. 在张量并行（world_size > 1）场景下，通过共享内存 + Event 把 rank 0
#      的调度指令广播给其他 worker 进程。
import math
import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from pathlib import Path

import torch
import torch.distributed as dist

from ..engine.sequence import Sequence
from ..layers.sampler import SamplerLayer
from ..models.llama3 import LlamaForCausalLM
from ..models.qwen3 import Qwen3ForCausalLM
from ..utils import *


class ModelRunner:
    """模型运行器：封装模型加载、KV 缓存分配、前向执行与多进程协调。

    rank 0 与 worker 进程共用同一个类：
      - rank 0（主引擎所在进程）：直接本地执行方法，同时通过 write_shm 把
        方法名和参数广播给 worker；
      - worker 进程：在 loop() 中阻塞等待共享内存里的指令，收到后调用对应
        方法（通常触发 NCCL 集合通信，与 rank 0 保持同步）。
    """

    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        """
        Args:
            config: 引擎配置字典（模型路径、并行度、显存利用率、分块大小等）。
            rank:   当前进程的全局序号，0 为主进程。
            event:  rank 0 收到的是所有 worker 的 Event 列表（用于逐个唤醒）；
                    worker 收到的是自己专属的那个 Event（用于阻塞等待指令）。
        """
        self.config = config
        # self.event = event
        # Split event storage by rank to avoid Union type issues
        # 注意 event 参数的类型是联合类型（单个 Event 或 Event 列表），
        # 这里按 rank 拆开存，避免后续代码里到处做类型判断。
        if isinstance(event, list):
            self.events: list[Event] = (
                event  # rank 0 gets the list of all worker events
            )
            self.event: Event | None = None  # rank 0 never uses a single event
        else:
            self.event: Event = event  # worker ranks get their own event
            self.events: list[Event] | None = None  # workers never use the events list

        # set distributed config
        # 读取分布式/推理相关的配置项
        self.block_size = config["block_size"]  # KV 缓存分块大小（每块容纳的 token 数）
        self.world_size = config["world_size"]  # 张量并行总进程数
        self.enforce_eager = config.get(
            "enforce_eager", False
        )  # True 则禁用 CUDA Graph

        self.rank = rank
        # 初始化 NCCL 进程组：所有进程必须用相同的 address、world_size，
        # 且各自带正确的 rank，才能彼此握手建联。
        dist.init_process_group(
            "nccl", "tcp://localhost:12345", world_size=config["world_size"], rank=rank
        )
        # 把后续的 CUDA 操作绑定到本 rank 对应的 GPU 上
        torch.cuda.set_device(rank)

        # set model
        # 根据模型路径的最后一段目录名匹配模型架构（本实现只支持 Qwen3 和 Llama3.2）
        path_str = self.config["model_name_or_path"]
        model_name = Path(path_str).name
        match model_name:
            case "Qwen3-0.6B":
                self.model = Qwen3ForCausalLM(
                    vocab_size=config["vocab_size"],
                    hidden_size=config["hidden_size"],
                    num_heads=config["num_heads"],
                    head_dim=config["head_dim"],
                    scale=config["scale"],
                    num_kv_heads=config["num_kv_heads"],
                    rms_norm_epsilon=config["rms_norm_epsilon"],
                    qkv_bias=config["qkv_bias"],
                    base=config["base"],
                    max_position=config["max_position"],
                    intermediate_size=config["intermediate_size"],
                    ffn_bias=config["ffn_bias"],
                    num_layers=config["num_layers"],
                    tie_word_embeddings=config["tie_word_embeddings"],
                    block_size=self.block_size,
                )
            case "Llama-3.2-1B-Instruct":
                self.model = LlamaForCausalLM(
                    vocab_size=config["vocab_size"],
                    hidden_size=config["hidden_size"],
                    head_dim=config["head_dim"],
                    num_qo_heads=config["num_qo_heads"],
                    num_kv_heads=config["num_kv_heads"],
                    has_attn_bias=config["has_attn_bias"],
                    rms_norm_epsilon=config["rms_norm_epsilon"],
                    rope_base=config["rope_base"],
                    max_position_embeddings=config["max_position_embeddings"],
                    intermediate_size=config["intermediate_size"],
                    ffn_bias=config["ffn_bias"],
                    num_layers=config["num_layers"],
                    block_size=self.block_size,
                    tie_word_embeddings=config["tie_word_embeddings"],
                )
            case _:
                raise Exception(f"Unsupported model: {config['model_name_or_path']}")

        # Load weights in GPU (model moved to GPU before loading weights)
        # 先把（随机初始化的）模型搬到目标 GPU，再加载权重，
        # 避免 CPU -> GPU 的额外搬运；张量并行下每个 rank 只持有分片后的部分。
        self.model = self.model.cuda(rank)

        # Load pretrained weights if model_name_or_path is provided
        # 从 checkpoint（safetensors / bin）读取预训练权重并写入模型分片
        if config.get("model_name_or_path"):
            from myvllm.utils.loader import load_weights_from_checkpoint

            load_weights_from_checkpoint(self.model, config["model_name_or_path"])

        # Load weights in CPU (move the model to GPU after loading weights)
        # self.model = self.model.cuda(rank)

        # 采样层：根据 temperature 对 logits 做 softmax 采样出下一个 token
        self.sampler = SamplerLayer()

        # Store default dtype before it's needed in allocate_kv_cache
        # 提前保存 PyTorch 全局默认 dtype：allocate_kv_cache 会按它计算 KV 块的字节数；
        # 之后 warmup / capture 过程会临时修改全局默认设置，退出前需用它恢复。
        self.default_dtype = torch.get_default_dtype()

        # Debug flag for first decode step
        self._first_decode = False

        # warm up model so that we know peak memory usage
        # ① 预热：用最大 batch 跑一遍空序列，让 PyTorch 的显存分配器把缓存
        #    arena 拉满，从而测出模型执行的峰值显存占用，供 KV 缓存分配参考。
        self.warmup_model()
        # allocate kv cache
        # ② 用剩余显存一次性分配尽可能大的 Paged KV 缓存池
        self.allocate_kv_cache()
        # capture cuda graph for decoding
        # ③ 非 eager 模式下，为各常见 batch size 捕获 decode 的 CUDA Graph
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 恢复全局默认 device / dtype，保证后续（如 Sampler 等模块）行为一致
        torch.set_default_device(f"cuda:{rank}")
        torch.set_default_dtype(self.default_dtype)

        # IMPORTANT: Set up shared memory and barrier AFTER all model initialization
        # This ensures both ranks complete warmup/allocation before rank 1 enters its event loop
        # 注意：共享内存与栅栏的搭建必须放在所有模型初始化完成之后，
        # 确保 rank 0/1 都完成 warmup 和缓存分配后，worker 才进入事件循环。
        if self.world_size > 1:
            # Synchronize before setting up shared memory
            # 第一道栅栏：所有 rank 对齐到"初始化完成"这一同步点
            dist.barrier()
            if self.rank == 0:
                # Try to clean up existing shared memory first
                # rank 0 负责创建共享内存；先清理上次运行残留的同名段，
                # 否则 create=True 会因名字已存在而报错。
                try:
                    old_shm = SharedMemory(name="myvllm")
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass  # Doesn't exist, which is fine
                # 创建 1MB 的共享内存段，用于 rank 0 -> workers 的指令广播
                self.shm = SharedMemory(name="myvllm", create=True, size=2**20)
                # Barrier to ensure rank 1 waits until shared memory is created
                # 第二道栅栏：保证 rank 1 打开共享内存时它一定已存在
                dist.barrier()
            else:
                # Wait for rank 0 to create shared memory
                # worker：先等 rank 0 创建完成，再以 attach 模式打开同一块共享内存
                dist.barrier()
                self.shm = SharedMemory(name="myvllm")
                # Don't call self.loop() here - let the spawning code handle it
                # Otherwise we'll be stuck in an infinite loop during __init__
                # 这里绝不能直接调用 self.loop()：那样 worker 会在 __init__
                # 中陷入死循环；事件循环由外层 spawn 代码显式启动。

    # only use read when rank != 0
    def read_shm(self):
        """worker 从共享内存读取一条指令（阻塞式）。

        流程：等待 rank 0 的 Event 通知 -> 按"前 4 字节小端序长度 + 后续
        pickle 载荷"的协议解析出 (method_name, args) -> 清除 Event。
        Returns:
            (method_name, args)：要调用的方法名及参数元组。
        """
        assert self.world_size > 1 and self.rank != 0, (
            "read_shm can only be called when world_size > 1 and rank != 0"
        )
        # 阻塞等待 rank 0 写入新指令并 set() 事件
        self.event.wait()
        assert self.shm.buf
        n = int.from_bytes(self.shm.buf[:4], "little")  # read length
        # 载荷格式为 pickle 序列化后的 (method_name, arg1, arg2, ...)
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        # 读完后清除事件，下次 wait 会重新阻塞，避免重复消费同一条指令
        self.event.clear()
        return method_name, args

    # only use write when rank == 0
    def write_shm(self, method_name: str, args: tuple):
        """rank 0 向共享内存写入一条指令，并唤醒所有 worker。

        协议：前 4 字节 = 载荷长度（小端 uint32），其后为 pickle 载荷
        (method_name, *args)。
        """
        assert self.world_size > 1 and self.rank == 0, (
            "write_shm can only be called when world_size > 1 and rank == 0"
        )
        # encode the length first
        # Flatten: (method_name, args) where args is a tuple -> (method_name, *args)
        # 把 (方法名, 参数元组) 拍平成 (方法名, 参数1, 参数2, ...) 再序列化
        data = pickle.dumps((method_name, *args))
        n = len(data)
        assert self.shm.buf
        # 先写长度，再写载荷（长度必须先就位，reader 才能正确切分）
        self.shm.buf[:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        assert self.events
        # 逐个 set 每个 worker 的 Event，唤醒它们去消费这条指令
        for event in self.events:
            event.set()

    # close shared memory, destroy process group, delete graphs
    def exit(self):
        """资源清理：关闭/销毁共享内存、释放 CUDA Graph、销毁进程组。"""
        # 共享内存：所有进程都要 close；只有 rank 0（创建者）负责 unlink 彻底释放
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        # CUDA Graph 及其预分配变量此时已不再使用，主动删除以释放显存
        if not self.enforce_eager:
            del self.graphs
            del self.graph_vars
        torch.cuda.synchronize()
        # Check if process group exists before destroying
        # 销毁 NCCL 进程组（带存在性检查，避免重复销毁报错）
        if dist.is_initialized():
            dist.destroy_process_group()

    # wait to read method and args from shared memory
    # execute the method with args
    # write results back to shared memory
    def loop(self):
        """worker 进程的事件主循环（仅 world_size > 1 且 rank != 0 时调用）。

        不断从共享内存取指令并执行，直到收到 "exit" 才退出。
        """
        assert self.world_size > 1 and self.rank != 0, (
            "loop can only be called when world_size > 1 and rank != 0"
        )
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)  # Unpack args when calling
            if method_name == "exit":
                # 收到退出指令：执行清理后结束循环，进程随之结束
                self.exit()
                break

    # will be called by both rank == 0 and rank != 0
    # given method name and args from shared memory
    # execute the method and return results
    def call(self, method_name: str, *args: dict):
        """统一的方法调用入口，兼容单机与多进程两种模式。

        - rank 0：先把指令广播到共享内存（worker 会同步执行同样的方法），
          然后本地执行并返回结果；
        - worker：由 loop() 调用，直接本地执行。
        """
        if self.world_size > 1 and self.rank == 0:  # will be called in main engine
            self.write_shm(method_name, args)
        # 反射拿到对应的成员方法并调用；未注册的方法名直接报错
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    # cleanup memory
    # compute max number of sequence based on max token and max model length
    # run empty sequence to warm up the model
    # clear memory
    def warmup_model(self):
        """模型预热：测出执行期的峰值显存占用，为 KV 缓存分配预留依据。

        做法：按 max_num_batch_tokens // max_model_length 估算最大可并发的
        序列数，构造等长的空序列（token 填 0）跑一次 prefill，然后清空缓存。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_tokens = self.config["max_num_batch_tokens"]
        max_model_length = self.config["max_model_length"]
        # 一批最多能装下几条满长序列
        batch_size = max_tokens // max_model_length
        seqs = [
            Sequence(
                token_ids=[0] * max_model_length, block_size=self.config["block_size"]
            )
            for _ in range(batch_size)
        ]
        # 用最大规模的 prefill 触发所有算子/显存分配路径
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    # allocate kv cache memory blocks for model
    def allocate_kv_cache(self):
        """分配 Paged Attention 的 KV 缓存池，并按层挂载到各 Attention 模块。

        步骤：
          1. 统计可用显存（扣除执行期峰值波动量）；
          2. 按每个 KV 块的真实字节数计算可容纳的块数 max_cached_blocks；
          3. 多卡时通过 all_reduce(MIN) 取各 rank 的最小值做全局同步，
             防止调度器分配超出某个 rank 容量的块数导致 OOM；
          4. 一次性分配 [2, num_layers, max_cached_blocks, block_size,
             num_kv_heads, head_dim] 的连续显存池，再按层切成视图挂到模型上。
        """
        # find all available memory
        # 查询当前 GPU 的空闲显存（free_mem 是驱动层空闲，不含 PyTorch 已缓存的）
        free_mem, total_mem = torch.cuda.mem_get_info()
        # 按 gpu_memory_utilization 比例划出可用于本引擎的部分（其余留给系统/其他进程）
        total_free_mem = free_mem * self.config["gpu_memory_utilization"]
        # 峰值显存占用（warmup 时统计得到）
        peak_mem_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # 当前实际占用的显存（模型权重等）
        current_mem_usage = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # reserve some room for peak memory usage during model execution
        # 峰值与当前的差值就是"执行期临时增量"，必须预留出来，否则推理时可能 OOM
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)

        # find parameters to compute kv cache size
        # 计算 KV 缓存形状所需的参数
        num_layers = self.config["num_layers"]
        num_kv_heads = (
            self.config["num_kv_heads"] // self.world_size
        )  # TP 时每个 rank 只存自己那几颗 KV head
        head_dim = (
            self.config["head_dim"]
            if "head_dim" in self.config
            else self.config["hidden_size"] // self.config["num_heads"]
        )

        # check whether the current free memory can hold at least one block
        # compute the actual byte required of one block
        # 一个 KV 块的字节数 = 块大小 × (K+V 两份) × 层数 × (每 rank 的)KV头数
        #                  × 头维度 × 每元素字节数
        block_bytes = (
            self.block_size
            * 2  # K 和 V 共用计数
            * num_layers
            * num_kv_heads
            * head_dim
            * self.default_dtype.itemsize
        )
        # 可分配的块数 = 可用显存 // 每块字节数（向下取整）
        num_available_kv_blocks = int(available_mem // block_bytes)
        # 至少要能装下一个块，否则根本无法启动推理
        assert num_available_kv_blocks >= 1, (
            f"Not enough memory to hold at least one block of KV cache on rank {self.rank}"
        )

        # Synchronize max_cached_blocks across all ranks.
        # Each rank independently computed num_available_kv_blocks from its own
        # free GPU memory. Ranks may differ slightly: rank-0 carries extra overhead
        # (NCCL buffers, process-group state) so it often has less free memory than
        # workers. Without sync, the scheduler (which runs only on rank-0) would use
        # rank-0's local value and could allocate more blocks than some rank can hold,
        # causing an OOM on that rank during KV cache writes.
        # 多卡时必须全局同步块数上限：每个 rank 各自算出的值可能不同
        # （rank 0 额外承担 NCCL 缓冲等开销，空闲显存通常更少）。若不做同步，
        # 只在 rank 0 上运行的调度器会按本地值分配超过其他 rank 容量的块，
        # 导致 worker 写 KV 缓存时 OOM。
        if self.world_size > 1:
            print(
                f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}"
            )
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks, dtype=torch.long, device=f"cuda:{self.rank}"
            )
            # all_reduce with MIN: every rank learns the most conservative limit,
            # i.e. the block count that even the most memory-constrained rank can serve.
            # This single agreed-upon value is then stored in config so the Scheduler
            # (initialized afterwards on rank-0) never allocates more blocks than any
            # rank can physically hold.
            # all_reduce(MIN)：所有 rank 对齐到"最保守"的上限——即显存最紧张的
            # rank 也能容纳的块数。该值写回 config，供之后在 rank 0 上初始化的
            # 调度器（Scheduler）使用，保证任何 rank 都不会被超额分配。
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config["max_cached_blocks"] = per_rank_max_blocks_tensor.item()
        else:
            # Single GPU: no cross-rank sync needed; use the local value directly.
            # 单卡无需跨进程同步，直接使用本地计算值
            self.config["max_cached_blocks"] = num_available_kv_blocks
        if self.rank == 0:
            print(
                f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}"
            )

        # 为模型分配尽可能大的 KV 缓存，而不是为每个序列单独分配。
        # 这是 Paged Attention（分页注意力）的关键：一个巨大的 KV 缓存池，被划分为多个块。
        # IMPORTANT: zeros() 来避免 NaN 垃圾值被 attention 读到（prefix cache 未命中的尾部位置仍会被 mask，但脏数据在某些 kernel 路径上会出问题）
        # 分配一整块连续显存
        # 维度含义：[0/1 表示 K/V, 层号, 物理块号, 块内 token 偏移, KV 头, 头维度]
        allocated_kv_cache = torch.zeros(
            2,
            self.config["num_layers"],
            self.config["max_cached_blocks"],
            self.block_size,
            num_kv_heads,
            head_dim,
            device=f"cuda:{self.rank}",
        )
        # 按 layer 切成视图挂到每个 Attention 模块上——不复制数据，只是同一块内存的不同偏移
        # 通过是否存在 k_cache/v_cache 属性来识别 Attention 层；
        # 切片只是创建视图（零拷贝），所有层共享同一个底层显存池
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                layer_id += 1

    # given seqs
    # prepare the data needed for a prefill forward pass
    # taking prefix cache into consideration:
    # input_ids, positions, cu_seqlens_q/k, slot_mapping (where to write new KV values), block_tables (where to read KV values)
    # cu_seqlens_q = [0, 3, 5, 9]
    #               │  │  │  │
    #               │  │  │  └─ end of seq3 (position 9)
    #               │  │  └──── end of seq2 (position 5)
    #               │  └─────── end of seq1 (position 3)
    #               └────────── start (position 0)
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        """准备 prefill 前向所需的全部输入张量，返回拼接后的 input_ids。

        prefill 是变长场景：一个 batch 里多条序列长度不一，所有新 token 被
        拼成一维长串，用 cu_seqlens_q/k 累积长度边界描述每条序列的起止
        （配合 flash_attn_varlen_func 做无 padding 的注意力计算）。

        前缀缓存（prefix cache）的影响：
          - q（query）：只包含"未被缓存的增量 token"；
          - k（key）：包含全部 token（缓存部分作为已写入的 KV 被读取）；
          - slot_mapping：只为增量 token 生成写入位置，且最后一块按
            last_block_num_tokens 截断（不满块不写满）。

        Returns:
            一维 int64 张量：本 batch 全部增量 token 拼接而成。
        """
        # length: sum of all input_ids after prefix cache
        # 增量 token 的总拼接列表（长度 = 各序列 q 之和）
        input_ids = []
        # length: sum of all input_ids after prefix cache
        # 每个增量 token 的写入 slot（全局逻辑槽号 = block_id * block_size + 块内偏移）
        slot_mappings = []
        # length: num_seqs
        # 每条序列本次要计算的 token 数（q 长度）
        seqlens_q = []
        # length: num_seqs
        # 每条序列的总长度（k 长度，包含缓存命中的前缀）
        seqlens_k = []
        # length: num_seqs + 1
        # q 的累积长度前缀和（首元素恒为 0），示例见方法 docstring 的 ASCII 图
        cu_seqlens_q = [0]
        # length: num_seqs + 1
        # k 的累积长度前缀和
        cu_seqlens_k = [0]
        # block_tables: num_seqs x num_blocks (padded)
        # 二维块表（行按最长序列用 -1 padding），用于读取已缓存的 KV
        block_tables = []
        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = seq.num_cached_tokens
            # 只发送缓存未命中的增量部分
            input_ids.extend(token_ids[num_cached_tokens:])
            seqlens_q.append(len(token_ids) - num_cached_tokens)
            seqlens_k.append(len(token_ids))
            # 维护累积长度前缀和
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
            if seq.block_table:
                # 只给 num_cached_blocks 之后的块生成 slot，且最后一块按 last_block_num_tokens 截断（不满块不写满）；
                # 如果存在前缀复用（cu_seqlens_q[-1] < cu_seqlens_k[-1]）才需要 pad 并下发 block_tables
                # 为增量 token 生成写入 slot：
                #   - 中间的完整块：整块 slot 全部写入；
                #   - 最后一块：只写到 last_block_num_tokens 为止。
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
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            # pad block_tables
            # 存在前缀复用：注意力 kernel 需要读取已缓存部分的 KV，
            # 因此要把每条序列的块表对齐到同一宽度（用 -1 padding）下发
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            for i, seq in enumerate(seqs):
                block_table = seq.block_table + [-1] * (
                    max_num_blocks - len(seq.block_table)
                )
                block_tables.append(block_table)
        # pin_memory + non_blocking：CPU 侧锁页内存异步拷贝，隐藏 H2D 传输延迟
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping_tensor = torch.tensor(
            slot_mappings, dtype=torch.long, pin_memory=True
        ).cuda(non_blocking=True)

        # 把本次前向需要的全部上下文注册进全局 context，
        # 后续 Attention 算子（attention kernel）会从这里取参数
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(
                cu_seqlens_q, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(
                cu_seqlens_k, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),  # kernel 优化用的上限：q 最大长度
            max_seqlen_k=max(seqlens_k),  # kernel 优化用的上限：k 最大长度
            slot_mapping=slot_mapping_tensor,  # 新 KV 的写入位置
            context_lens=None,  # prefill 不需要（长度隐含在 cu_seqlens_k 中）
            block_tables=torch.tensor(
                block_tables, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            if block_tables
            else None,  # 仅存在前缀复用时才下发
        )
        return input_ids

    # prepare input data for decoding
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        """准备 decode 前向所需的输入张量，返回 input_ids。

        decode 每条序列只前向一个新 token，因此 batch 形状固定为
        (batch_size,)，天然适合 CUDA Graph 复用。

        关键张量分工：
          - slot_mapping（写）：新算出的 K/V 要写到哪个全局槽位
            （= 最后一块物理块号 × block_size + 块内偏移）；
          - context_lens（读）：每条序列已处理的 token 总数，即读 KV 的边界；
          - block_tables（读）：(batch, max_num_blocks) 二维表，逻辑块号 ->
            物理块号的映射，配合块内偏移即可定位任意 token 的 KV。
        """
        input_ids = []
        context_lens = []  # 读的时候的序列长度边界。
        slot_mappings = []  # 写用。全局逻辑 slot 编号 = block_id * block_size + 块内偏移，把新算出的 K/V 写到哪。
        block_tables = []  # 读用。(batch, max_num_blocks) 的二维表，第 i 行的第 j 项是该序列第 j 个逻辑块对应的物理块号，解引用后加上块内偏移即可定位。

        # 一条序列只处理一个新 token，所以 slot 就是最后一块的最后一个位置
        for seq in seqs:
            input_ids.append(seq.last_token)  # 只取最后一个 token 参与本次前向
            context_lens.append(len(seq))  # 已有的上下文长度（读 KV 的边界）
            slot_mappings.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )  # 最后一块的最后一个位置
        # 块表按最长序列对齐（短序列用 -1 padding，kernel 会跳过）
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            block_table = seq.block_table + [-1] * (
                max_num_blocks - len(seq.block_table)
            )
            block_tables.append(block_table)
        # 同样用锁页内存异步传输
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(
            non_blocking=True
        )
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,  # decode 是定长 batch，不需要变长边界
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(
                slot_mappings, dtype=torch.long, pin_memory=True
            ).cuda(non_blocking=True),
            context_lens=torch.tensor(
                context_lens, dtype=torch.long, pin_memory=True
            ).cuda(non_blocking=True),
            block_tables=torch.tensor(
                block_tables, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            if block_tables
            else None,
        )
        return input_ids

    # prepare the temperature
    def prepare_sample(self, seqs: list[Sequence]) -> None:
        """收集每条序列的采样温度，返回 (batch,) 的 float32 GPU 张量。"""
        return torch.tensor(
            [seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)

    # when prefilling, directly compute model forward + logits
    # when decoding, use cuda graph execution to speed up
    # allocate input_ids, positions, slot_mapping, context_lens, block_tables, outputs
    # into graph_variable, and then replay the graph
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        """执行一次模型前向，返回 logits。

        - prefill / enforce_eager：直接以 eager 模式跑前向；
        - decode（已捕获 CUDA Graph）：把真实输入 copy_ 进预分配的图变量，
          再 replay 对应 batch size 的图，省去逐 kernel 的启动开销。
        """
        if is_prefill or self.enforce_eager:
            # For varlen prefill, keep input_ids as 1D (concatenated tokens)
            # Do NOT unsqueeze - flash_attn_varlen_func expects 1D input with cu_seqlens
            # prefill 走变长路径：input_ids 保持一维拼接（不要 unsqueeze），
            # 配合 set_context 注册的 cu_seqlens 交给 flash_attn_varlen_func
            hidden_states = self.model(input_ids)
            # 在拼接的 hidden_states 上计算最后一个位置的 logits
            logits = self.model.compute_logits(hidden_states)
        else:
            bs = input_ids.size(0)
            context = get_context()

            # 运行时选最小的够用 graph，用 copy_ 灌数据再 replay()；
            # slot_mapping 先 fill_(-1) 是为了让 padding 的 batch 位置不写缓存
            # （配合 kernel 里的 if slot_idx == -1: return）：
            # 挑选"能容纳当前 bs 的最小"图（图按 [1,2,4,8,16,32,...] 预先捕获）
            graph = self.graphs[next(bs_ for bs_ in self.graphs if bs_ >= bs)]
            vars = self.graph_vars
            # copy input data into graph variables
            # 把真实输入写入捕获时固定地址的图变量（copy_ 不产生新分配，
            # 这是 CUDA Graph 能复用的前提）
            vars["input_ids"][:bs].copy_(input_ids)
            # 先整体填 -1：padding 位置的 slot 为 -1，attention kernel
            # 检测到 -1 会跳过写入，避免垃圾 KV 污染缓存
            vars["slot_mapping"][:bs].fill_(-1)
            vars["slot_mapping"][:bs].copy_(context.slot_mapping)
            # context_lens / block_tables 的 padding 区置零（0 长度上下文不会被读到）
            vars["context_lens"].zero_()
            vars["context_lens"][:bs].copy_(context.context_lens)
            vars["block_tables"][:bs, : context.block_tables.size(1)] = (
                context.block_tables
            )
            # replay the graph
            # 重放捕获好的 kernel 序列，输出直接落在预分配的 outputs 上
            graph.replay()
            # logits 的计算放在图外：这样不需要为 vocab_size 大小的输出捕获额外显存
            logits = self.model.compute_logits(vars["outputs"][:bs])

        return logits

    # prepare prefill
    # prepare sample
    # run model
    # sample logits
    # reset context
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """一次完整的前向-采样流水线：prepare -> run_model -> sample。

        Returns:
            rank 0 返回采样出的 token id 列表（长度 = batch 大小）；
            其他 rank 返回 None（采样只需在 rank 0 做一次）。
        """
        # 1. 按阶段准备输入张量并注册 context
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        # 2. 前向计算 logits（decode 时走 CUDA Graph）
        logits = self.run_model(input_ids, is_prefill)
        # only sample when rank == 0
        # 3. 只有 rank 0 需要采样（结果由上层调度器消费并广播）
        token_ids = None
        if self.rank == 0:
            token_ids = self.sampler(logits, self.prepare_sample(seqs))
        # 4. 清理全局 context，防止下一次前向误用旧参数
        reset_context()
        return token_ids

    # capture the CUDA graph:
    # pre-allocation at maximum sizes: allocated onece and reuse for all graphs
    # capture for different common batch sizes: [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    # with torch.cuda.graph(graph, self.graph_pool):
    #        run model() and exact sequence of CUDA kernels for running self.model() will be captured
    # (later use graph.replay() to run the captured graph)
    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        """为 decode 阶段捕获各 batch size 的 CUDA Graph。

        CUDA Graph 要求所有张量的地址在捕获与重放时完全一致，因此：
          1. 按 max_bs × max_num_blocks 预分配图变量（input_ids、slot_mapping、
             context_lens、block_tables、outputs），各图共用同一组地址；
          2. 对每个常见 batch size 各捕获一份图（存储在 self.graphs）；
          3. 所有图共享同一个内存池（graph_pool），避免显存碎片化；
          4. 推理时用 copy_ 把真实数据灌入图变量，再 graph.replay()。
        """
        max_bs = self.config["max_num_sequences"]
        max_len = self.config["max_model_length"]
        # 最长序列需要的物理块数（向上取整）
        max_num_blocks = math.ceil(max_len / self.block_size)
        # for decoding, input is always of shape (batch_size, 1)
        # decode 的输入永远是每个序列一个 token，形状 (batch_size,)
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f"cuda:{self.rank}")

        # decode 时 KV 相关的三个输入张量必须在 capture 前预分配好固定地址，否则 graph 无法复用。
        # 所以 capture_cudagraph 按 max_bs × max_num_blocks 预分配 slot_mapping / context_lens / block_tables，
        # 然后对 [1,2,4,8] + range(16, max_bs+1, 16) 每个 batch size 各捕获一份：
        # for paged attention
        # where to write new KV values in the cache
        # 新 K/V 的写入槽位（batch 维按 max_bs 预分配）
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f"cuda:{self.rank}")
        # how many tokens each sequence has processed
        # 每条序列已处理的 token 数（读 KV 的边界）
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f"cuda:{self.rank}")
        # where to read KV values in the cache
        # 逻辑块 -> 物理块的映射表（batch × max_num_blocks 预分配）
        block_tables = torch.zeros(
            max_bs, max_num_blocks, dtype=torch.int32, device=f"cuda:{self.rank}"
        )
        # output hidden states（logits 由 compute_logits 在 graph 外计算，故这里是 hidden_size 而非 vocab_size）
        # 图的输出缓冲：每条序列最后一个 token 的 hidden states
        outputs = torch.zeros(
            max_bs,
            self.config["hidden_size"],
            dtype=next(self.model.parameters()).dtype,
            device=f"cuda:{self.rank}",
        )

        # graphs to be captured for different batch sizes
        # 捕获的 batch size 档位：小 bs 密集（1/2/4/8），大 bs 每 16 一档
        batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        # 所有图共享一个内存池：第一张图捕获结束后取其 pool() 给后续图复用
        graph_pool = None

        # 从大到小捕获：先捕获最大的图，其内存池确定后再复用给小图，
        # 保证小图不会与大图在池中冲突
        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            # 先以 eager 方式跑一次（warmup），让 cuBLAS/cuDNN 完成工作区
            # 分配等副作用；同时用真实形状注册 context
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )
            outputs[:batch_size] = self.model(input_ids[:batch_size])

            # 真正的捕获阶段：with 块内的 kernel 序列被录进 graph，不再真正执行
            with torch.cuda.graph(graph, graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
            # pool() 必须在 capture 成功结束后才能调用，故放在 with 之外
            # 第一张图捕获成功后，把它的工作区内存池共享给后续所有图
            if graph_pool is None:
                graph_pool = graph.pool()
            # store the captured graph
            self.graphs[batch_size] = graph

            # make sure that the capture is done before resetting and next capture
            # 每张图捕获完成后同步一次，再清理 context 进入下一次捕获
            torch.cuda.synchronize()
            reset_context()

        # 保存图变量：运行时 run_model() 通过 copy_ 把真实输入写进这些
        # 预分配张量，然后 replay 对应的图
        self.graph_vars = {
            "input_ids": input_ids,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "outputs": outputs,
        }
