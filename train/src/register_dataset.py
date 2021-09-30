from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog


def registerDataset(datasetName, jsonPath, imageDirPath, numOfClasses):
    register_coco_instances(datasetName, {}, jsonPath, imageDirPath)
    MetadataCatalog.get(datasetName).thing_classes = numOfClasses