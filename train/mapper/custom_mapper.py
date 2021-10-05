from train.mapper.base_mapper import BaseMapper
import detectron2.data.transforms as T

class CustomMapper(BaseMapper):

	def __init__(self):
			super().__init__()
			self._addTransForm(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))
			self._addTransForm(T.RandomBrightness(0.8, 1.8))
			self._addTransForm(T.RandomContrast(0.6, 1.3))
