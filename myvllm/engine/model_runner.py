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
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        self.config = config
        # self.event = event
        # Split event storage by rank to avoid Union type issues
        if event is Event:
            self.events: list[Event] = (
                event  # rank 0 gets the list of all worker events
            )
            self.event: Event | None = None  # rank 0 never uses a single event
        elif event is list[Event]:
            self.event: Event = event  # worker ranks get their own event
            self.events: list[Event] | None = None  # workers never use the events list

        # set distributed config
        self.block_size = config["block_size"]
        self.world_size = config["world_size"]
        self.enforce_eager = config.get("enforce_eager", False)

        self.rank = rank
        dist.init_process_group(
            "nccl", "tcp://localhost:12345", world_size=config["world_size"], rank=rank
        )
        torch.cuda.set_device(rank)

        # set model
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
        self.model = self.model.cuda(rank)

        # Load pretrained weights if model_name_or_path is provided
        if config.get("model_name_or_path"):
            from myvllm.utils.loader import load_weights_from_checkpoint

            load_weights_from_checkpoint(self.model, config["model_name_or_path"])

        # Load weights in CPU (move the model to GPU after loading weights)
        # self.model = self.model.cuda(rank)

        self.sampler = SamplerLayer()

        # Store default dtype before it's needed in allocate_kv_cache
        self.default_dtype = torch.get_default_dtype()

        # Debug flag for first decode step
        self._first_decode = False

        # warm up model so that we know peak memory usage
        self.warmup_model()
        # allocate kv cache
        self.allocate_kv_cache()
        # capture cuda graph for decoding
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device(f"cuda:{rank}")
        torch.set_default_dtype(self.default_dtype)

        # IMPORTANT: Set up shared memory and barrier AFTER all model initialization
        # This ensures both ranks complete warmup/allocation before rank 1 enters its event loop
        if self.world_size > 1:
            # Synchronize before setting up shared memory
            dist.barrier()
            if self.rank == 0:
                # Try to clean up existing shared memory first
                try:
                    old_shm = SharedMemory(name="myvllm")
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass  # Doesn't exist, which is fine
                self.shm = SharedMemory(name="myvllm", create=True, size=2**20)
                # Barrier to ensure rank 1 waits until shared memory is created
                dist.barrier()
            else:
                # Wait for rank 0 to create shared memory
                dist.barrier()
                self.shm = SharedMemory(name="myvllm")
                # Don't call self.loop() here - let the spawning code handle it
                # Otherwise we'll be stuck in an infinite loop during __init__

    # only use read when rank != 0
    def read_shm(self):
        assert self.world_size > 1 and self.rank != 0, (
            "read_shm can only be called when world_size > 1 and rank != 0"
        )
        self.event.wait()
        assert self.shm.buf
        n = int.from_bytes(self.shm.buf[:4], "little")  # read length
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    # only use write when rank == 0
    def write_shm(self, method_name: str, args: tuple):
        assert self.world_size > 1 and self.rank == 0, (
            "write_shm can only be called when world_size > 1 and rank == 0"
        )
        # encode the length first
        # Flatten: (method_name, args) where args is a tuple -> (method_name, *args)
        data = pickle.dumps((method_name, *args))
        n = len(data)
        assert self.shm.buf
        self.shm.buf[:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        assert self.events
        for event in self.events:
            event.set()

    # close shared memory, destroy process group, delete graphs
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs
            del self.graph_vars
        torch.cuda.synchronize()
        # Check if process group exists before destroying
        if dist.is_initialized():
            dist.destroy_process_group()

    # wait to read method and args from shared memory
    # execute the method with args
    # write results back to shared memory
    def loop(self):
        assert self.world_size > 1 and self.rank != 0, (
            "loop can only be called when world_size > 1 and rank != 0"
        )
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)  # Unpack args when calling
            if method_name == "exit":
                self.exit()
                break

    # will be called by both rank == 0 and rank != 0
    # given method name and args from shared memory
    # execute the method and return results
    def call(self, method_name: str, *args: dict):
        if self.world_size > 1 and self.rank == 0:  # will be called in main engine
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    # cleanup memory
    # compute max number of sequence based on max token and max model length
    # run empty sequence to warm up the model
    # clear memory
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_tokens = self.config["max_num_batch_tokens"]
        max_model_length = self.config["max_model_length"]
        batch_size = max_tokens // max_model_length
        seqs = [
            Sequence(
                token_ids=[0] * max_model_length, block_size=self.config["block_size"]
            )
            for _ in range(batch_size)
        ]
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    # allocate kv cache memory blocks for model
    def allocate_kv_cache(self):
        # find all available memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        total_free_mem = free_mem * self.config["gpu_memory_utilization"]
        peak_mem_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current_mem_usage = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # reserve some room for peak memory usage during model execution
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)

        # find parameters to compute kv cache size
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
        # compute the actual byte required of each block
        block_bytes = (
            self.block_size
            * 2  # K 和 V 共用计数
            * num_layers
            * num_kv_heads
            * head_dim
            * self.default_dtype.itemsize
        )
        num_available_kv_blocks = int(available_mem // block_bytes)
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
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config["max_cached_blocks"] = per_rank_max_blocks_tensor.item()
        else:
            # Single GPU: no cross-rank sync needed; use the local value directly.
            self.config["max_cached_blocks"] = num_available_kv_blocks
        if self.rank == 0:
            print(
                f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}"
            )

        # 为模型分配尽可能大的 KV 缓存，而不是为每个序列单独分配。
        # 这是 Paged Attention（分页注意力）的关键：一个巨大的 KV 缓存池，被划分为多个块。
        # IMPORTANT: zeros() 来避免 NaN 垃圾值被 attention 读到（prefix cache 未命中的尾部位置仍会被 mask，但脏数据在某些 kernel 路径上会出问题）
        # 分配一整块连续显存
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
        # length: sum of all input_ids after prefix cache
        input_ids = []
        # length: sum of all input_ids after prefix cache
        slot_mappings = []
        # length: num_seqs
        seqlens_q = []
        # length: num_seqs
        seqlens_k = []
        # length: num_seqs + 1
        cu_seqlens_q = [0]
        # length: num_seqs + 1
        cu_seqlens_k = [0]
        # block_tables: num_seqs x num_blocks (padded)
        block_tables = []
        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = seq.num_cached_tokens
            input_ids.extend(token_ids[num_cached_tokens:])
            seqlens_q.append(len(token_ids) - num_cached_tokens)
            seqlens_k.append(len(token_ids))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
            if seq.block_table:
                # 只给 num_cached_blocks 之后的块生成 slot，且最后一块按 last_block_num_tokens 截断（不满块不写满）；
                # 如果存在前缀复用（cu_seqlens_q[-1] < cu_seqlens_k[-1]）才需要 pad 并下发 block_tables
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
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            for i, seq in enumerate(seqs):
                block_table = seq.block_table + [-1] * (
                    max_num_blocks - len(seq.block_table)
                )
                block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping_tensor = torch.tensor(
            slot_mappings, dtype=torch.long, pin_memory=True
        ).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(
                cu_seqlens_q, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(
                cu_seqlens_k, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=torch.tensor(
                block_tables, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            if block_tables
            else None,
        )
        return input_ids

    # prepare input data for decoding
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        context_lens = []  # 读的时候的序列长度边界。
        slot_mappings = []  # 写用。全局逻辑 slot 编号 = block_id * block_size + 块内偏移，把新算出的 K/V 写到哪。
        block_tables = []  # 读用。(batch, max_num_blocks) 的二维表，第 i 行的第 j 项是该序列第 j 个逻辑块对应的物理块号，解引用后加上块内偏移即可定位。

        # 一条序列只处理一个新 token，所以 slot 就是最后一块的最后一个位置
        for seq in seqs:
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))
            slot_mappings.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            block_table = seq.block_table + [-1] * (
                max_num_blocks - len(seq.block_table)
            )
            block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(
            non_blocking=True
        )
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
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
        return torch.tensor(
            [seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)

    # when prefilling, directly compute model forward + logits
    # when decoding, use cuda graph execution to speed up
    # allocate input_ids, positions, slot_mapping, context_lens, block_tables, outputs
    # into graph_variable, and then replay the graph
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill or self.enforce_eager:
            # For varlen prefill, keep input_ids as 1D (concatenated tokens)
            # Do NOT unsqueeze - flash_attn_varlen_func expects 1D input with cu_seqlens
            hidden_states = self.model(input_ids)
            logits = self.model.compute_logits(hidden_states)
        else:
            bs = input_ids.size(0)
            context = get_context()

            # 运行时选最小的够用 graph，用 copy_ 灌数据再 replay()；
            # slot_mapping 先 fill_(-1) 是为了让 padding 的 batch 位置不写缓存
            # （配合 kernel 里的 if slot_idx == -1: return）：
            graph = self.graphs[next(bs_ for bs_ in self.graphs if bs_ >= bs)]
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
            # replay the graph
            graph.replay()
            logits = self.model.compute_logits(vars["outputs"][:bs])

        return logits

    # prepare prefill
    # prepare sample
    # run model
    # sample logits
    # reset context
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, is_prefill)
        # only sample when rank == 0
        token_ids = None
        if self.rank == 0:
            token_ids = self.sampler(logits, self.prepare_sample(seqs))
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
        max_bs = self.config["max_num_seqs"]
        max_len = self.config["max_model_length"]
        max_num_blocks = math.ceil(max_len / self.block_size)
        # for decoding, input is always of shape (batch_size, 1)
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f"cuda:{self.rank}")

        # decode 时 KV 相关的三个输入张量必须在 capture 前预分配好固定地址，否则 graph 无法复用。
        # 所以 capture_cudagraph 按 max_bs × max_num_blocks 预分配 slot_mapping / context_lens / block_tables，
        # 然后对 [1,2,4,8] + range(16, max_bs+1, 16) 每个 batch size 各捕获一份：
        # for paged attention
        # where to write new KV values in the cache
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f"cuda:{self.rank}")
        # how many tokens each sequence has processed
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f"cuda:{self.rank}")
        # where to read KV values in the cache
        block_tables = torch.zeros(
            max_bs, max_num_blocks, dtype=torch.int32, device=f"cuda:{self.rank}"
        )
        # output logits
        outputs = torch.zeros(
            max_bs, self.config["vocab_size"], device=f"cuda:{self.rank}"
        )

        # graphs to be captured for different batch sizes
        batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        graph_pool = None

        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
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

            with torch.cuda.graph(graph, graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
                if graph_pool is None:
                    graph_pool = graph.pool()
            # store the captured graph
            self.graphs[batch_size] = graph

            # make sure that the capture is done before resetting and next capture
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
