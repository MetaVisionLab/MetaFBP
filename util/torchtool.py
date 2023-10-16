import pickle
import shutil
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict, Iterable
from typing import Union, Tuple

import torch
import torch.nn as nn


def set_requries_grad(model, requries_grad: bool):
    if isinstance(model, nn.Module):
        for p in model.parameters():
            p.requries_grad = requries_grad
    elif isinstance(model, Iterable):
        for p in model:
            p.requries_grad = requries_grad
    elif isinstance(model, torch.Tensor):
        model.requries_grad = requries_grad
    else:
        raise RuntimeError(f'Unknown type({type(model)}) to set grad')


def load_checkpoint(fpath):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')

    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else 'cpu'

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint


def load_pretrained_weights(model, state_dict):
    r"""Load pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        state_dict (dict): path to pretrained weights.
    """
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    else:
        state_dict = state_dict

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights  cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'
        )
    else:
        print(
            'Successfully loaded pretrained weights'
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                    format(discarded_layers)
            )


class SigmoidMapper(nn.Module):
    def __init__(self, score: Union[Tuple[float], float]):
        super().__init__()
        if isinstance(score, tuple):
            assert len(score) == 2
            self.min_score = score[0]
            self.max_score = score[1]
        self.min_score = 0
        self.max_score = score
        assert self.min_score < self.max_score

    def forward(self, x):
        return self.min_score + torch.sigmoid(x) * (self.max_score - self.min_score)
