from train.mapper.base_mapper import DummyAlbuMapper
import albumentations as A

class CustomMapper(DummyAlbuMapper):

	def __init__(self):
			super().__init__()
			self._addTransForm(A.Flip(p=0.5))
			self._addTransForm(A.RandomRotate90(p=0.5))

			self._addTransForm(A.RandomResizedCrop(height=1024, width=1024, scale=(0.5, 1.0), p=0.5))
			self._addTransForm(A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.5))
			self._addTransForm(A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5))
			self._addTransForm(A.GaussNoise(p=0.3))

			self._addTransForm(A.OneOf())


# dict(
# type='OneOf',
# transforms=[
# 	dict(type='Flip',p=1.0),
# 	dict(type='RandomRotate90',p=1.0)
# ],
# p=0.5),
# dict(type='RandomResizedCrop',height=1024, width=1024, scale=(0.5, 1.0), p=0.5),
# dict(type='RandomBrightnessContrast',brightness_limit=0.1, contrast_limit=0.15, p=0.5),
# dict(type='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5),
# dict(type='GaussNoise', p=0.3),
# dict(
# type='OneOf',
# transforms=[
# 	dict(type='Blur', p=1.0),
# 	dict(type='GaussianBlur', p=1.0),
# 	dict(type='MedianBlur', blur_limit=5, p=1.0),
# 	dict(type='MotionBlur', p=1.0)
# ],
# p=0.1)