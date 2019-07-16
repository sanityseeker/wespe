import os

import torch

def save_state(models, optimizers, save_dir, epoch, device):
  path_to_save_file = os.path.join(save_dir, 'checkpoint.tar')
  save_config = dict()
  save_config['epoch'] = epoch
  for (name, model) in models:
    save_config[f'{name}_state_dict'] = model.cpu().state_dict()
    model.to(device)
  for (name, optimizer) in optimizers:
    save_config[f'{name}_state_dict'] = optimizer.state_dict()
  torch.save(save_config, path_to_save_file)


def restore_state(models, optimizers, load_dir):
  path_to_load_file = os.path.join(load_dir, 'checkpoint.tar')
  checkpoint = torch.load(path_to_load_file)
  for (name, model) in models:
    model.load_state_dict(checkpoint[f'{name}_state_dict'])
  for (name, optimizer) in optimizers:
    optimizer.load_state_dict(checkpoint[f'{name}_state_dict'])
