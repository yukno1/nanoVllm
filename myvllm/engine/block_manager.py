from collections import deque

import numpy as np
import xxhash

from myvllm.engine.sequence import Sequence


class Block:
    __slots__ = ("block_id", "hash", "ref_count", "token_ids")

    def __init__(self, block_id):
        self.block_id = block_id
        self.hash = -1
        self.ref_count = 0
        self.token_ids = []

    def update(self, h: int, token_ids: list[int]):
        self.hash = h
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1
        self.ref_count = 0
        self.token_ids = []


class BlockManager:
    __slots__ = (
        "block_size",
        "blocks",
        # 三类队列
        "free_block_ids",  # deque
        "hash_to_block_id",  # 前缀缓存哈希表
        "used_block_ids",  # set
    )

    def __init__(self, num_blocks: int, block_size: int):
        # block_size: number of tokens per block
        self.block_size: int = block_size
        # list of all blocks
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash to block id: this is for prefix caching
        self.hash_to_block_id: dict[int, int] = {}
        # free block ids
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # used block ids
        self.used_block_ids: set[int] = set()

    # given token_ids, compute the hash value
    # use prefix_hash_value to compute the hash in a context-sensitive way
    # 前缀缓存哈希是链式的：块 hash = xxhash(前一块 hash || 本块 token_ids)，只有满块才计算，不满的块 hash 固定为 -1（不参与复用）
    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, "little"))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    # whether we can allocate a block for this sequence
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        h = -1
        # 逐块查表
        for i in range(seq.num_blocks):
            no_cache_found = False

            token_ids = seq.block(i)
            # only compute hash for full blocks, always -1 for partial blocks
            h = (
                self.compute_hash(token_ids=token_ids, prefix_hash_value=h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = self.hash_to_block_id.get(h, -1)

            # 如果缓存未命中或哈希碰撞
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                no_cache_found = True

            if not no_cache_found:
                # 命中就 ref_count += 1 共享物理块并累加 seq.num_cached_tokens，未命中就从 free 队列取新块。
                # update sequence information
                seq.num_cached_tokens += self.block_size  # which == len(token_ids)
                # update block information, considering the edge case that the block is not allocated yet but with hash code
                if block_id not in self.used_block_ids:
                    block = self._allocate_block(block_id)
                else:
                    # update block information
                    block = self.blocks[self.hash_to_block_id[h]]
                    block.ref_count += 1
            else:
                # cache miss
                block = self._allocate_block(self.free_block_ids[0])
                block.update(h=h, token_ids=token_ids)
                if h != -1:
                    self.hash_to_block_id[h] = block.block_id
            seq.block_table.append(block.block_id)

    def deallocate(self, seq: Sequence) -> None:
        """
        走引用计数，归零才真正回收到 free 队列，所以多个请求共享的相同 prompt 前缀不会被提前释放。\n
        NOTE 块被回收时不清 hash 表，这就是前缀缓存（跨请求命中）能生效的原因；_deallocate_block 只清 token_ids 用于下次碰撞校验。
        """

        # update block information
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # update sequence information
        seq.block_table = []
        seq.num_cached_tokens = 0

    # this is to check whether we can append tokens to this sequence
    # when that token would require allocating a new block.
    def can_append(self, seq: Sequence) -> bool:
        if seq.num_tokens % self.block_size == 0:
            return len(self.free_block_ids) > 0
        return True

    def append(self, seq: Sequence) -> None:
        """
        处理解码增长：块刚好写满时固化 hash；下一 token 需要新块时（num_tokens % block_size == 1）才真正分配。\n
        这是将 token 追加到该序列的实际操作。
        当新 token 被添加到序列信息后，会调用此函数，但此时 GPU 中尚未为其分配任何显存块。
        """
        block_tables = seq.block_table
        last_block_for_seq_id = block_tables[-1]

        # if the last block is now full, compute hash
        if seq.num_tokens % self.block_size == 0:
            h = self.compute_hash(
                token_ids=seq.block(seq.num_blocks - 1),
                prefix_hash_value=-1
                if len(block_tables) == 1
                else self.blocks[block_tables[-2]].hash,
            )
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
            self.hash_to_block_id[h] = block.block_id
        # if one new block is needed
        elif seq.num_tokens % self.block_size == 1:
            # Previous block should be finalized
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_block(self.free_block_ids[0])
            block_tables.append(block.block_id)
        # else, do nothing
        else:
            assert last_block_for_seq_id in self.used_block_ids, (
                "Last block should be allocated"
            )
            assert self.blocks[last_block_for_seq_id].hash == -1, (
                "Last block should be partial block with hash -1"
            )

    # move this block to used list
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.blocks[block_id]
        block.token_ids = []
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
