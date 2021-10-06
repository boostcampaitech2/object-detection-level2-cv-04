from typing import Optional
from detectron2.engine import DefaultTrainer
from detectron2.engine.train_loop import HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
import os
from detectron2.engine.defaults import hooks
from fvcore.nn.precise_bn import get_bn_modules

from tqdm import tqdm

from detectron2.utils import comm
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, get_event_storage
import torch

class TQDMHook(HookBase):
  def __init__(self, step) -> None:
      super().__init__()
      self.step = step
  
  def after_step(self):
    if (self.trainer.iter) % self.step == 0:
      self.trainer.showTQDM.update(self.step)


# def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
#     """
#     Build an optimizer from config.
#     """
#     params = get_default_optimizer_params(
#         model,
#         base_lr=cfg.SOLVER.BASE_LR,
#         weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
#         bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
#         weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
#     )
#     return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
#         params,
#         lr=cfg.SOLVER.BASE_LR,
#         momentum=cfg.SOLVER.MOMENTUM,
#         nesterov=cfg.SOLVER.NESTEROV,
#         weight_decay=cfg.SOLVER.WEIGHT_DECAY,
#     )


class BaseTrainer(DefaultTrainer):

  outputEvalDir = ""
  mapper = None
  sampler = None

  def __init__(self, cfg):
    # self.build_optimizer = build_optimizer
    self.showTQDM = tqdm(range(cfg.SOLVER.MAX_ITER))
    super().__init__(cfg)

  @classmethod
  def build_train_loader(cls, cfg, sampler=sampler):
    return build_detection_train_loader(
    cfg, mapper = cls.mapper, sampler = cls.sampler
    )
    
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
      os.makedirs(cls.outputEvalDir,exist_ok=True)
      output_folder = cls.outputEvalDir
        
    return COCOEvaluator(dataset_name, cfg, False, output_folder)


  def build_hooks(self):
      """
      Build a list of default hooks, including timing, evaluation,
      checkpointing, lr scheduling, precise BN, writing events.

      Returns:
          list[HookBase]:
      """
      cfg = self.cfg.clone()
      cfg.defrost()
      cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

      ret = [
          hooks.IterationTimer(),
          hooks.LRScheduler(),
          hooks.PreciseBN(
              # Run at the same freq as (but before) evaluation.
              cfg.TEST.EVAL_PERIOD,
              self.model,
              # Build a new data loader to not affect training
              self.build_train_loader(cfg),
              cfg.TEST.PRECISE_BN.NUM_ITER,
          )
          if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
          else None,
      ]

      # Do PreciseBN before checkpointer, because it updates the model and need to
      # be saved by checkpointer.
      # This is not always the best: if checkpointing has a different frequency,
      # some checkpoints may have more precise statistics than others.
      if comm.is_main_process():
          ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

      def test_and_save_results():
          self._last_eval_results = self.test(self.cfg, self.model)
          return self._last_eval_results

      # Do evaluation after checkpointer, because then if it fails,
      # we can use the saved checkpoint to debug.
      ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

      if comm.is_main_process():
          # Here the default print/log frequency of each writer is used.
          # run writers in the end, so that evaluation metrics are written
          writerList = [
                        CustomMetricPrinter(self.showTQDM,self.cfg.SOLVER.MAX_ITER),
                        JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                        TensorboardXWriter(self.cfg.OUTPUT_DIR),
                      ]
                      

          ret.append(hooks.PeriodicWriter(writerList, period=10))

          ret.append(TQDMHook(step=1))

      return ret


class CustomMetricPrinter(CommonMetricPrinter):
  def __init__(self, tqdmModule ,max_iter: Optional[int] = None, window_size: int = 20):
      super().__init__(max_iter=max_iter, window_size=window_size)
      self.tqdmModule = tqdmModule
  

  def write(self):
    storage = get_event_storage()
    iteration = storage.iter
    if iteration == self._max_iter:
        return

    try:
        lr = "{:.5g}".format(storage.history("lr").latest())
    except KeyError:
        lr = "N/A"


    showDict = {"lr":lr}
    lossTuple = [(k, f"{v.median(self._window_size):.4g}") for k, v in storage.histories().items() if "loss" in k]
    for k,v in lossTuple:
      showDict[k] = v

    self.tqdmModule.set_postfix(showDict)