import argparse
from config.load_config import makeDictByConfig

class ConfigArgParser(argparse.ArgumentParser):
	
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		
		cfgDict = makeDictByConfig()
		print(cfgDict)
		self._addArgParser()

	def _addArgParser(self):
		# print(dict(self.cfg))
		pass
'''
	python -m train --seed 4
'''
