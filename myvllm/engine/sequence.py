from itertools import count


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self) -> None:
        pass

    def __len__(self):
        pass

    def __getitem__(self, key):
        pass

    @property
    def is_finished(self):
        pass

    @property
    def num_block(self):
        pass
