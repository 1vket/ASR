import math
import logging
import defect

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class TransformerModel(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.config = config

    self.e_pos_emb = nn.Parameter(
      torch.zeros(1, config.block_size, config.n_mel))
    self.ln = nn.LayerNorm(config.n_mel)
    self.dense = nn.Linear(config.n_mel, config.d_model)

    self.tok_emb = nn.Embedding(
      config.vocab_size, config.d_model, padding_idx=config.pad_idx)
    self.d_pos_emb = nn.Parameter(
      torch.zeros(1, config.block_size, config.d_model))

    self.transformer = nn.Transformer(
      d_model = config.d_model,
      num_encoder_layers = config.n_layer,
      num_decoder_layers = config.n_layer,
      dim_feedforward = config.d_model * 4,
      activation = 'gelu',
      batch_first=True)

    self.out = nn.Linear(config.d_model, config.vocab_size)

    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if isinstance(module, nn.Linear) and module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
      torch.nn.init.zeros_(module.bias)
      torch.nn.init.ones_(module.weight)
    elif isinstance(module, TransformerModel):
      torch.nn.init.normal_(module.e_pos_emb, mean=0.0, std=0.02)
      torch.nn.init.normal_(module.d_pos_emb, mean=0.0, std=0.02)

  def forward(self, mfcc, src, tgt=None):
    # encoder
    b, et, em = mfcc.size()

    e_position_emb = self.e_pos_emb[:, :et, :]
    
    ein = self.dense(self.ln(mfcc + e_position_emb))

    # decoder
    b, dt = src.size()

    d_position_emb = self.d_pos_emb[:, :dt, :]

    din = self.tok_emb(src) + d_position_emb

    # transformer
    device = 'cpu'
    if torch.cuda.is_available():
      device = torch.cuda.current_device()
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(dt).to(device)
    out = self.transformer(ein, din, tgt_mask=tgt_mask)

    logits = self.out(out)

    loss = None
    if tgt is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1),
       ignore_index=self.config.pad_idx)

    return logits, loss

  def configure_optimizers(self, train_config):
    
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (
      torch.nn.Linear, 
    )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in self.named_modules():
      for pn, p in m.named_parameters():
        fpn = '%s.%s' % (mn, pn) if mn else pn

        if pn.endswith('bias'):
          no_decay.add(fpn)
        elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
          decay.add(fpn)
        elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
          no_decay.add(fpn)
        elif pn.endswith('in_proj_weight'):
          decay.add(fpn)

    no_decay.add('e_pos_emb')
    no_decay.add('d_pos_emb')

    param_dict = {pn: p for pn, p in self.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, \
      "parameters not into either decay/no_decay sets!"

    optim_groups = [
      {"params": [param_dict[pn] for pn in sorted(list(decay))], \
        "weight_decay": train_config.weight_decay},
      {"params": [param_dict[pn] for pn in sorted(list(no_decay))], \
        "weight_decay": 0.0}
    ]

    optimizer = torch.optim.AdamW(
      optim_groups, lr=train_config.learning_rate, betas=train_config.betas)

    return optimizer




