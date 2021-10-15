from typing import Optional
from detectron2.data.samplers import TrainingSampler

class CustomSampler(TrainingSampler):
	def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
			super().__init__(size, shuffle=shuffle, seed=seed)