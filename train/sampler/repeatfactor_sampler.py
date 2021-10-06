from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.data.common import DatasetFromList
from detectron2.data.build import get_detection_dataset_dicts
from numpy.core.fromnumeric import repeat

class RepeatfactorSampler(RepeatFactorTrainingSampler):
	"""
	Similar to TrainingSampler, but a sample may appear more times than others based on its "repeat factor". 
	This is suitable for training on class imbalanced datasets like LVIS.
	"""
	repeat_factors = None	
	threshold = None
	def __init__(self, size:int, shuffle=True, seed=None):
		super().__init__(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(self.repeat_factors,self.threshold), 
			shuffle=shuffle, seed=seed)