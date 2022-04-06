
import math
import logging

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)

class Trainer:
  def __init__(self, model, train_dataset, test_dataset, config):
    self.model = model
    self.train_dataset = train_dataset
    self.test_dataset = test_dataset
    self.config = config

    self.device = 'cpu'
    if torch.cuda.is_available():
      self.device = torch.cuda.current_device()
      self.model = torch.nn.DataParallel(self.model).to(self.device)

    print(f"current device {self.device}")

  def save_checkpoint(self):
    raw_model = self.model.module if hasattr(self.model,"module")  \
      else self.model

    logger.info("saving %s", self.config.ckpt_path)
    torch.save(raw_model.state_dict(), self.config.ckpt_path)

  def train(self):
    model, config = self.model, self.config
    raw_model = model.module if hasattr(self.model, "module") else model
    optimizer = raw_model.configure_optimizers(config)

    def run_epoch(loader, is_train):
      model.train(is_train)

      losses = []
      pbar = tqdm(enumerate(loader)) if is_train \
        else enumerate(loader)

      for it, (mfcc, x, y, ml, sl) in pbar:
        
        ml = max(ml)
        mfcc = mfcc.to(self.device)
        mfcc = mfcc[:, :ml, :]

        sl = max(sl)
        x = x[:, :sl].to(self.device)
        y = y[:, :sl].to(self.device)

        with torch.set_grad_enabled(is_train):
          logits, loss = model(mfcc, x, y)
          loss = loss.mean()
          losses.append(loss.item())

        if is_train:
          model.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_norm_clip)

          optimizer.step()

          if config.lr_decay:
            self.tokens += (y >= 1).sum()
            if self.tokens < config.warmup_tokens:
              lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
            else:
              progress = float(self.tokens - config.warmup_tokens) / \
                float(max(1, config.final_tokens - config.warmup_tokens))
              lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = config.learning_rate * lr_mult
            for param_group in optimizer.param_groups:
              param_group['lr'] = lr

          else:
            lr = config.learning_rate

          pbar.set_description(
            f"epoch {epoch+1} iter {it}:"
            +f"train loss {loss.item():.5f}. lr {lr:e}")

      if not is_train:
        test_loss = float(np.mean(losses))
        logger.info("test loss: %f", test_loss)
        return test_loss
    
    best_loss = float('inf')
    self.tokens = 0

    train_loader = DataLoader(
      self.train_dataset,
      shuffle=True,
      pin_memory=True,
      batch_size=config.batch_size,
      num_workers=config.num_workers
    )

    if self.test_dataset is not None:
      test_loader = DataLoader(
        self.test_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers
      )

    for epoch in range(config.max_epochs):
      run_epoch(train_loader, is_train=True)
      if self.test_dataset is not None:
        test_loss = run_epoch(test_loader, is_train=False)

      good_model = self.test_dataset is None or test_loss < best_loss
      if self.config.ckpt_path is not None and good_model:
        best_loss = test_loss
        self.save_checkpoint()

