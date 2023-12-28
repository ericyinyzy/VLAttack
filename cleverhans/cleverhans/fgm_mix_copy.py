"""The Fast Gradient Method attack."""
import numpy as np
import torch
import torch.nn as nn

from cleverhans.torch.utils import optimize_linear
import torch.nn.functional as F
from torchvision import transforms
def input_diversity(input_tensor):
  # input_tensor=copy.deepcopy(input)
  image_height=input_tensor.shape[-2]
  image_width=input_tensor.shape[-1]
  newh=int(np.random.uniform(image_height-32,image_height))
  neww=int((newh/image_height)*image_width)
  # print('neww',neww,newh,input_tensor[:,:].shape)
  # exit()
  rescaled = transforms.functional.resize(input_tensor, (newh,neww))
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
  # print('padded',padded[0,0,:32,:32],padded.shape)
  # exit()
  return padded
# np.random.seed(0)
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
    ls=None,
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
    # if y is None:
    #     # Using model predictions as ground truth to avoid label leaking
    #     _, y = torch.max(model_fn(x), 1)

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
    #     print(aa)
    #     out=model_fn(input_diversity(x))
    # else:
    # patch_weight, _ = patch_fn(ori_x+x,ori_x)
    # patch_weight,opt_mask=patch_fn(ori_x+x,y,ori_x)
    # perts=ori_x+x
    out = model_fn(x)

    # exit()

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
    #########################feature loss
    feat_len=min(out[2].shape[1],y[2].shape[1])
    out[2]=out[2][:,:feat_len,:]
    y[2]=y[2][:,:feat_len,:]
    #############################
    # print(out[2].shape)
    # exit()
    # print(out[1].shape,out[2].shape,y[1].shape,y[2].shape)
    # exit()
    # patch_weight=patch_weight[:feat_len]

    # feat_len_1 = min(y[0].shape[1], y[2].shape[1])
    # out[2] = out[2][:, :feat_len_1, :]
    # y[2] = y[2][:, :feat_len, :]
    # print(torch.norm(x-ori_x))
    # print(F.pairwise_distance(out[1], y[1],p=2).shape,out[1].shape,y[1].shape,l2(out[1],y[1]),out[2].shape,y[2].shape,F.pairwise_distance(out[2], y[2],p=2),torch.sum(F.pairwise_distance(out[2], y[2],p=2),1))
    # exit()
    # print(torch.sum(torch.std(out[1], dim=-1)+torch.sum(torch.std(out[2], dim=-1),1)/(out[2].shape[1])))
    # exit()
    #torch.sum(F.pairwise_distance(out[1], y[1],p=2)+ torch.sum(F.pairwise_distance(out[2], y[2],p=2), 1)))
    # print(torch.sum((-cos(out[1], y[1]) + torch.sum(-cos1(out[2], y[2]), 1))))
    # print(y[0].shape,y[2].shape)#,torch.mul(cos1(out[2], y[2]),y[0]).shape)
    # loss_attack_patch=torch.sum((-cos(out[1], y[1]) + torch.sum(-cos1(out[2], y[2]), 1)))
    # patch_weight=gen_patch_weight(loss_attack_patch)
    # exit()
    # print(type(y),type(out))
    # exit()
    # exit()
    # print(out[0].shape,y[0].shape)

    # exit()
    if ls==1:
        loss=(torch.sum((-cos(out[1], y[1]) + torch.sum(-cos1(out[2], y[2]), 1))))
    elif ls==0:
        loss = (
                   # torch.dot((-cos(out[1], y[1]) + ((-cos1(out[2], y[2]).mm(y[-2].unsqueeze(-1))).squeeze(-1))),y[-1])
                # torch.sum(-cos(out[1], y[1])+((-cos1(out[2], y[2]).mm(y[-2].unsqueeze(-1))).squeeze(-1)))
            #        torch.sum((-cos(out[1], y[1]) + torch.dot(-cos1(out[2], y[2]), y[-2])))
            # -l2(out[1], y[1])
           # torch.sum(-torch.std(out[1], dim=-1)+torch.sum(-torch.std(out[2], dim=-1),1)/(out[2].shape[1]))
           #  torch.sum(F.pairwise_distance(out[1], y[1],p=2)+ torch.sum(F.pairwise_distance(out[2], y[2],p=2), 1))
           #  torch.sum((-cos(out[1], y[1]) + torch.sum(torch.mul(-cos1(out[2], y[2]),patch_weight), 1)))#+0.1*(torch.sum(out[3]))
           #  torch.sum((-cos(out[1], y[1]) + torch.sum(-cos1(out[2], y[2]), 1)))/(out[2].shape[0]*out[2].shape[1])+0.1*F.cross_entropy(out[0].view(-1, 30522),y[0].view(-1),ignore_index = -100)#+0.1*F.cross_entropy(out[3], y[3])
            F.cross_entropy(out[0].view(-1, 30522),y[0].view(-1),ignore_index = -100)#+F.cross_entropy(out[1], y[1])
           # torch.sum((cos(out[1], y[1]) ))
            # +
            # torch.norm(x - ori_x)
            # torch.sum((cos(out[1], y[1]) + torch.sum(cos1(out[2], y[2]), 1)))
            # torch.sum(cos(out[1], y[1]))
            #        torch.dot((-cos(out[1], y[1])+torch.sum(-cos1(out[2], y[2]),1)),y[-1])
            #     torch.dot(-cos(out[1], y[1]), y[-1])
            #     torch.dot(-cos(out[1], y[1]), y[-1])+ torch.dot(-cos(out[2], y[2]),y[-2])#+torch.sum(-cos(out[1][:-120], y[1][:-120]))*0.6+torch.sum(-cos(out[1][-120:-50], y[1][-120:-50]))*0.45+torch.sum(-cos(out[1][-50:-20], y[1][-50:-20]))*0.3#+((torch.sum(model_fn(x)[2])/20)+(1-torch.sum(model_fn(x)[1])/50))*3#+torch.sum(-cos(model_fn(x)[1], y[1]))*0.5#
            #     100*(kl_loss(obj, target)+kl_loss(target,obj)) #+ torch.dot(-cos(out[1], y[1]),y[-1])# + torch.sum(
            # -cos(out[1][-120:-50], y[1][-120:-50])) * 0.45 + torch.sum(-cos(out[1][-50:-20], y[1][-50:-20])) * 0.3
            #     -cos(model_fn(x)[0], y[0])#torch.dot(-cos(model_fn(x)[1], y[1]),y[-1])#torch.sum(-cos(model_fn(x)[1][:-40], y[1][:-40]))#torch.sum(-cos(model_fn(x)[1], y[1]))*400
        )
    # else:
    #     loss = torch.sum((-cos(out[1], y[1]) + torch.sum(-cos1(out[2], y[2]), 1)))/(out[2].shape[0]*out[2].shape[1])\
    #            +0.1*F.cross_entropy(out[0].view(-1, 30522),y[0].view(-1),ignore_index = -100)
    #     for synom_list in y[3]:
    #         loss=loss+0.1*F.cross_entropy(out[0].view(-1, 30522), synom_list[0].view(-1), ignore_index=-100)
    # print('loss',loss,torch.sum(-cos1(out[2], y[2]), 1))
  ########################
    # print('loss',)
    # exit()
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    loss.requires_grad_(True).backward()
    # print('aaaaa')
    # loss1.requires_grad_(True).backward()

    # loss.requires_grad_(True).backward()
    # print(model_fn(x)[-1].requires_grad)
    # exit()
    # print(x.grad.shape)
    # print(torch.where(x.grad==0)[0].shape)
    # exit()
    # exit()
    # print(x.grad.shape,opt_mask.shape)
    # grad=torch.mul(opt_mask,x.grad)
    # print(torch.where(grad==0)[0].shape)
    # exit()
    # print('xxxxxxxx')
    # print(perts.grad[])
    optimal_perturbation = optimize_linear(x.grad,eps , norm)
    # print('yyyyyyyyy')
    # print('x_grad',x.grad.shape)
    # exit()
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
    return adv_x,loss
