"""The Fast Gradient Method attack."""
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Resize,InterpolationMode

from cleverhans.torch.utils import optimize_linear
import torch.nn.functional as F
from torchvision import transforms
def input_diversity(input_tensor):
  # print(input_tensor.shape)
  # input_tensor=copy.deepcopy(input)
  image_height=input_tensor.shape[-2]
  image_width=input_tensor.shape[-1]
  newh=int(np.random.uniform(image_height-32,image_height))
  neww=int((newh/image_height)*image_width)
  rescaled = transforms.functional.resize(input_tensor, (newh,neww),InterpolationMode.NEAREST)
  # rescaled=F.resize(input_tensor,(1,3,newh, neww))
  h_rem = image_height - newh
  w_rem = image_width - neww
  pad_top = int(np.random.uniform( 0, h_rem))
  pad_bottom = h_rem - pad_top
  pad_left = int(np.random.uniform( 0, w_rem))
  pad_right = w_rem - pad_left
  padding = (
      pad_left, pad_right,  # 前面填充1个单位，后面填充两个单位，输入的最后一个维度则增加1+2个单位，成为8
      pad_top, pad_bottom,
  )
  padded = F.pad(rescaled, padding)
  if padded.shape[-1]!= input_tensor.shape[-1] or padded.shape[-2]!= input_tensor.shape[-2]:
      raise ValueError
  # print('aaa',newh,neww,pad_top,pad_bottom,pad_left,pad_right)
  # print('padded',padded[0,0,:32,:32],padded.shape)
  # exit()
  # print('padded',padded.shape)
  # exit()
  # print(padded.shape)

  return padded

def fast_gradient_method(
    model_fn,
    x,
    eps,
    norm,
    ori_x,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
    mean_tensors=None,
):
    """
    PyTorch implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)
    out = model_fn(x)
    f=[]
    f_adv=[]

    if len(out[0])==12:
        f_adv=y[0]
        f=out[0]
        loss = 0
        for i in range(12):
            wts_good = f_adv[i] < mean_tensors[i]
            wts_good = wts_good.float()
            wts_bad = f_adv[i] >= mean_tensors[i]
            wts_bad = wts_bad.float()
            loss += torch.log(torch.norm(
                wts_good * (f[i]) / (f[i].shape[0] * f[i].shape[1])) ** 2 / 2)
            loss -= torch.log(torch.norm(
                wts_bad * (f[i]) / (f[i].shape[0] * f[i].shape[1])) ** 2 / 2)
        loss = loss / 12
    else:
        for i, (feats, feats_ori) in enumerate(zip(out[0], y[0])):
            f.append(feats.view(1, 256 * (2 ** i), -1).permute(0, 2, 1))
            f_adv.append(feats_ori.view(1, 256 * (2 ** i), -1).permute(0, 2, 1))
        loss = 0
        for i in range(4):
            wts_good = f_adv[i] < mean_tensors[i]
            wts_good = wts_good.float()
            wts_bad = f_adv[i] >= mean_tensors[i]
            wts_bad = wts_bad.float()
            loss += torch.log(torch.norm(
                wts_good * (f[i]) / (f[i].shape[0] * f[i].shape[1])) ** 2 / 2)
            loss -= torch.log(torch.norm(
                wts_bad * (f[i]) / (f[i].shape[0] * f[i].shape[1])) ** 2 / 2)
        loss = loss / 4
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    loss.requires_grad_(True).backward()
    optimal_perturbation = optimize_linear(x.grad, eps, norm)
    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x,loss
