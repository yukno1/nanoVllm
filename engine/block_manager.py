class Block:
    def __init__(self, block_id) -> None:
        self.block_id = block_id

    def update(self):
        pass

    def reset(self):
        pass


class BlockManager:
    def __init__(self, num_block: int, block_size: int) -> None:
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_block)]
