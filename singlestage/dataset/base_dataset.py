import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from singlestage.src.augmentation.base_augmentation import transform


from singlestage.src.dataset_src.cocojson_to_dataframe import jsonToDataframe

class BaseDataset(Dataset):
  
  def __init__(self, jsonPath, imageRootPath):
    super().__init__()

    self.transformList = []
    self.imageRootPath = imageRootPath

    self.dataFrame = jsonToDataframe(jsonPath)
    
    self.fileNames = self.dataFrame['file_name'].tolist()
    # self.labels = torch.LongTensor(dataFrame['category_id'])
    # self.bboxs = torch.FloatTensor(dataFrame['bbox'])

  def __getitem__(self, index):
    # 이미지, 박스들, 라벨, difficulties를 반환함
    path = os.path.join(self.imageRootPath,self.fileNames[index])
    image = Image.open(path)
    matchFrame = self.dataFrame[self.dataFrame["image_id"]==index]
    box = torch.FloatTensor(matchFrame['bbox'].tolist())
    labels =  torch.LongTensor(matchFrame['category_id'].tolist())
    # transform도 해야됨
    image, box, labels, _ = transform(image, box, labels, None, "TRAIN")
    #모든 이미지 사이즈 1024x1024
    return image, box, labels


  def __len__(self):
    return len(self.dataFrame['image_id'].unique())

  def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        # difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            # difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels # tensor (N, 3, 300, 300), 3 lists of N tensors each