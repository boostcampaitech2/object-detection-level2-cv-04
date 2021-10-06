import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
import torch
import copy

class BaseMapper:

    def __init__(self):
        self.transformList = []

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict['file_name'], format='BGR')

        image, transforms = T.apply_transform_gens(self.transformList, image)

        dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))   

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop('annotations')
            if obj.get('iscrowd', 0) == 0
        ]

        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict['instances'] = utils.filter_empty_instances(instances)

        return dataset_dict
    
    def _addTransForm(self, transform):
        self.transformList.append(transform)