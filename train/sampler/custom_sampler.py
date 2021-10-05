from typing import Optional
from detectron2.data.samplers import *

class CustomSampler(TrainingSampler):
	def __init__(self, size: int = 4871, shuffle: bool = True, seed: Optional[int] = None):
			super().__init__(size, shuffle=shuffle, seed=seed)