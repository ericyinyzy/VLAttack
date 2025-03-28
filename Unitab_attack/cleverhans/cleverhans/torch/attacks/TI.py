"""The MI Projected Gradient Descent attack."""
import numpy as np
import torch
from torchvision import transforms
from cleverhans.torch.attacks.TI_FGSM import TI_fast_gradient_method
from cleverhans.torch.utils import clip_eta
import copy
import torch.nn.functional as F
import time as tt

def TI_projected_gradient_descent(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    ori_x=None,
    time=None,
    targeted=False,
    rand_init=True,
    rand_minmax=None,
    sanity_checks=True,
):
    """
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to False. or the
    Madry et al. (2017) method if rand_init is set to True.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations.
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
    :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
    :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
              which the random perturbation on x was drawn. Effective only when rand_init is
              True. Default equals to eps.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if eps_iter < 0:
        raise ValueError(
            "eps_iter must be greater than or equal to 0, got {} instead".format(
                eps_iter
            )
        )
    if eps_iter == 0:
        return x

    assert eps_iter <= eps, (eps_iter, eps)
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

    # Initialize loop variables
    if time==0:
        rand_init=False
    else:
        rand_init=False
    # print(rand_init)rand _init
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)

    else:
        eta = torch.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    # print('eta',eta.requires_grad)
    # if time==0:
    #     print('eta', eta[0,:10,0,0])
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    i = 0
    loss_list=[]
    # loss_last=loss_ini
    # step=0
    ori_x=ori_x.requires_grad_(False)
    noise=copy.deepcopy(eta)
    # print('time0', tt.time())
    while i < nb_iter:
        adv_x,noise = TI_fast_gradient_method(
            model_fn,
            adv_x,
            eps_iter,
            norm,
            noise,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
        )
        # if float(loss.detach().cpu().numpy())-loss_last<24:
        #     step+=1
        # else:
        #     loss_last=float(loss.detach().cpu().numpy())
        #     step=0
        #
        # if step>=max_step:
        #     return adv_x,loss_last
        # Clipping perturbation eta to norm norm ball
        eta = adv_x - ori_x
        eta = clip_eta(eta, norm, eps)
        adv_x = ori_x + eta
        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
        i += 1
    # print('end',end)
    # print('time1',tt.time())
    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast clip_min and clip_max to x.dtype?
        asserts.append(eps + clip_min <= clip_max)
    # print(asserts)
    asserts=[i.cpu() for i in asserts if i is not True]
    # exit()
    # print('time2',tt.time())
    if sanity_checks:
        assert np.all(asserts)
    return adv_x,loss_list


# def gkern(kernlen=21, nsig=3):
#   """Returns a 2D Gaussian kernel array."""
#   import scipy.stats as st
#
#   x = np.linspace(-nsig, nsig, kernlen)
#   kern1d = st.norm.pdf(x)
#   kernel_raw = np.outer(kern1d, kern1d)
#   kernel = kernel_raw / kernel_raw.sum()
#   return kernel
# kernel = gkern(15, 3).astype(np.float32)
#     stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
#     stack_kernel = torch.tensor(np.expand_dims(stack_kernel, 3))
#     stack_kernel=stack_kernel.permute(2,3,0,1).cuda()
#     # print(stack_kernel[:,0,0,0])
#     noise = x.grad
#     noise = noise / torch.norm(noise,p=1)
#     # print(noise.shape)
#     noise = F.conv2d(noise, stack_kernel,bias=None, stride=1, groups=3,padding=7)
#     # print(noise.shape)
#     # exit()
#     noise = lst_noise + noise
#     optimal_perturbation=torch.sign(noise)
#     optimal_perturbation = eps * optimal_perturbation