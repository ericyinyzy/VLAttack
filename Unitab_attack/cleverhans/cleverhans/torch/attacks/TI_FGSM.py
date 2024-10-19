"""The Fast Gradient Method attack."""
import numpy as np
import torch
import torch.nn as nn

from cleverhans.torch.utils import optimize_linear
import torch.nn.functional as F
from torchvision import transforms
def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel
# np.random.seed(0)
def TI_fast_gradient_method(
    model_fn,
    x,
    eps,
    norm,
    lst_noise,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
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

    # Compute loss
    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss = loss_fn(model_fn(x), y)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos1 = nn.CosineSimilarity(dim=2, eps=1e-6)
    l2 = nn.MSELoss()
    kl_loss=nn.KLDivLoss(log_target=True)
    m = nn.Softmax(dim=-1)

    # print(m(y[0]).shape,m(y[0]))
    # print(cos(model_fn(x)[1][:-10], y[1][:-10]).shape)
    # print(-cos(model_fn(x)[0], y[0]))
    # print(torch.sum(model_fn(x)[1]/50))
    # print(torch.sum(model_fn(x)[2]/20))

    # exit()
    # exit()
    # print(cos(model_fn(x)[1], y[1]).shape,torch.sum(-cos(model_fn(x)[1], y[1])))
    # print(cos(model_fn(x)[1], y[1]).shape)
    # print(torch.sum(-cos(model_fn(x)[1], y[1]))*100)
    # exit()
    # aa=np.random.uniform(0,1)
    # if aa<0.7:
    #     # print(aa)
    #     # torch.use_deterministic_algorithms(False)
    #     out=model_fn(input_diversity(x))
    # else:
    out = model_fn(x)
    # target=m(y[0]/0.001)
    # obj=m(out[0]/0.001)
    # print(kl_loss(obj, target)+kl_loss(target,obj))
    # print(l2(out[0], y[0]))
    # print(y[-2].unsqueeze(-1).shape)
    # print((cos(out[1], y[1])+((cos1(out[2], y[2]).mm(y[-2].unsqueeze(-1))).squeeze(-1))).shape)
    # print(m(out[-1]).shape,m(y[1]).shape,out[-1].shape,m(y[-1]).shape,torch.sum(m(y[-1])[0][0]))
    # print(kl_loss(m(out[-1]), m(y[1])).shape,kl_loss(m(out[-1]), m(y[1])),kl_loss(m(out[-1]).view(-1,768), m(y[1]).view(-1,768)),m(y[1]).view(-1,768).shape)
    # print(torch.dot(-cos1(out[2], y[2]), y[-2]).shape)
    # exit()
    # print(torch.sum(kl_loss(m(out[-1]), m(y[1])) + 5*kl_loss(m(out[-1]), m(y[-1]))))

    # print(m(out[-1]).shape,m(y[1]).shape,m(y[-1]).shape)
    # xx=m(out[-1])
    # print((-cos(out[1], y[1])+torch.sum(-cos1(out[2], y[2]),1)).shape)
    # print(torch.sum(xx[0]),torch.sum(xx[-1]))
    # print((kl_loss(m(out[-1]), m(y[1])) + 5*kl_loss(m(out[-1]),m(y[-1]))))
    # print((-cos(out[1], y[1]) + ((-cos1(out[2], y[2]).mm(y[-2].unsqueeze(-1))).squeeze(-1))).shape)
    # print((-cos(out[1], y[1]) + torch.sum(-cos1(out[2], y[2]), 1)).shape,torch.sum(-cos1(out[2], y[2]), 1).shape,cos1(out[2], y[2]).shape)
    # exit()
    # Co-attack loss
    # loss=(
    #         torch.sum(kl_loss(m(out[-1]), m(y[1])) + 5*kl_loss(m(out[-1]), m(y[-1])))
    #
    #
    #
    # )
    # print(out[-1].shape,y[-1].shape,y[0].shape)
    # exit()
    # if out[-1].shape[1] != y[-1].shape[1] or out[-1].shape[1] != y[0].shape[1]:
    #     print('shape not aligned')
    # feat_len = min(out[-1].shape[1], y[-1].shape[1],y[0].shape[1])
    # out[-1] = out[-1][:, :feat_len, :]
    # y[-1] = y[-1][:, :feat_len, :]
    # y[0] = y[0][:, :feat_len, :]
    # loss = (
    #         torch.sum((5*torch.sum(-cos1(out[-1], y[0]), 1) + torch.sum(-cos1(out[-1], y[-1]), 1)))
    #     )
    #VL- loss
    ###############
    # # print(out[2].shape[1], y[2].shape[1])

    feat_len = min(out[1].shape[1], y[1].shape[1])
    out[1] = out[1][:, :feat_len, :]
    y[1] = y[1][:, :feat_len, :]
    f=[]
    f_adv=[]
    # print(torch.sum(-cos1(out[0][0], out[0][0]), 1).shape)
    #
    # for x,y in zip(out[0],y[0]):
    #     print(x.shape,y.shape)
    # exit()

    # res_f0,resf_1,res_f2,res_f3=out[0][0],out[0][1],out[0][2],out[0][3]
    for i,(feats,feats_ori) in enumerate(zip(out[0],y[0])):
        f.append(feats.view(1,256*(2**i),-1).permute(0,2,1))
        f_adv.append(feats_ori.view(1,256*(2**i),-1).permute(0,2,1))
    loss = (
        torch.sum(torch.sum(-cos1(f[0], f_adv[0]), 1) + torch.sum(-cos1(f[1], f_adv[1]), 1) + torch.sum(
            -cos1(f[2], f_adv[2]), 1) + torch.sum(-cos1(f[3], f_adv[3]), 1) +torch.sum(-cos1(out[1], y[1]), 1))  # +0.1*(torch.sum(out[3]))
    )
    # print(torch.sum(-cos1(out[0][0].float(), y[0][0].float()), 1),torch.sum(-cos1(out[0][1].float(), y[0][1].float()), 1),torch.sum(-cos1(out[0][2].float(), y[0][2].float()), 1))
    # print(cos(out[0][1][:,0].float(), y[0][1][:,0].float()))
    # exit()
    # print(torch.sum(-cos1(out[1].float(), y[1].float()), 1))
    # exit()
    # loss = (
    #     torch.sum(torch.sum(-cos1(out[0][0], y[0][0]), 1)+torch.sum(-cos1(out[0][1], y[0][1]), 1)+torch.sum(-cos1(out[0][2], y[0][2]), 1)+torch.sum(-cos1(out[1], y[1]), 1))#+0.1*(torch.sum(out[3]))
    # )
  ########################
    # print('loss',loss,torch.sum(-cos1(out[0][0].float(), y[0][0].float()), 1),torch.sum(-cos1(out[0][1].float(), y[0][1].float()), 1),torch.sum(-cos1(out[0][2].float(), y[0][2].float()), 1),torch.sum(-cos1(out[1].float(), y[1].float()), 1))

    # x=x.float()
    # print('loss', loss, x.dtype)
    # exit()
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    loss.requires_grad_(True).backward()
    kernel = gkern(15, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = torch.tensor(np.expand_dims(stack_kernel, 3))
    stack_kernel=stack_kernel.permute(2,3,0,1).cuda().float()
    # print(stack_kernel[:,0,0,0])
    noise = x.grad
    noise = noise / torch.norm(noise,p=1)
    # print(noise.shape)
    noise = F.conv2d(noise, stack_kernel,bias=None, stride=1, groups=3,padding=7)
    # print(noise.shape)
    # exit()
    noise = lst_noise + noise
    optimal_perturbation=torch.sign(noise)
    optimal_perturbation = eps * optimal_perturbation
    # print()
    # print('optimal',torch.max(optimal_perturbation),torch.min(optimal_perturbation))

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
    # print(torch.max(adv_x-x),torch.min(adv_x-x))
    return adv_x,noise
