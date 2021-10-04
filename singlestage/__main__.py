from config.load_config import makeDictByConfig
from singlestage.dataset.base_dataset import BaseDataset
from singlestage.model.model import *
# from src.utils import gra
import time

import torch

grad_clip= None
config = makeDictByConfig()

decay_lr_to = 0.1

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  start_epoch = 0
  trainDataset = BaseDataset(config.path.train_json, config.path.image_root)
  train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=config.hyperparam.ims_per_batch, shuffle=True,
                                               collate_fn=trainDataset.collate_fn, 
                                               num_workers=config.general.num_workers,
                                               pin_memory=True)

  biases = list()
  not_biases = list()
  lr = config.hyperparam.base_lr
  momentum = 0.9
  weight_decay = config.hyperparam.gamma
  optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

  model = SSD300(n_classes=config.general.roi_num_classes).to(device)
  criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

  epochs = config.hyperparam.max_iter // (len(trainDataset) // 32)
  decay_lr_at = [it // (len(trainDataset) // 32) for it in config.hyperparam.steps]

    # Epochs
  for epoch in range(start_epoch, epochs):

      # Decay learning rate at particular epochs
      if epoch in decay_lr_at:
          adjust_learning_rate(optimizer, decay_lr_to)

      # One epoch's training
      train(train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch)

      # Save checkpoint
      save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
        
        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % 30 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == "__main__":
	main()