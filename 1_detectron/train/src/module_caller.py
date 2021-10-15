import os
from importlib import import_module

def _refineModuleName(name):
	barIndex = name.find('_')
	return name[0].upper() + name[1:barIndex]+name[barIndex+1].upper()+name[barIndex+2:]
	
def getModule(parentName,name):
	return getattr(import_module(f"train.{parentName}.{name}"), _refineModuleName(name))


