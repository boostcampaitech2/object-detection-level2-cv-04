from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog


def registerDataset(datasetName, jsonPath, imageDirPath, numOfClasses, istrain):
    register_coco_instances(datasetName, {}, jsonPath, imageDirPath)
    if istrain:
        MetadataCatalog.get(datasetName).thing_classes = numOfClasses
