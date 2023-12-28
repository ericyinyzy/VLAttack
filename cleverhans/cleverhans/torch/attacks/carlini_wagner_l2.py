"""The CarliniWagnerL2 attack."""
import torch
import torch.nn as nn


INF = float("inf")


def carlini_wagner_l2(
    model_fn,
    model_fn_compare,
    x,
    n_classes=None,
    y=None,
    ori_x=None,
    targeted=False,
    lr=5e-3,
    confidence=0,
    clip_min=0,
    clip_max=1,
    initial_const=1,
    binary_search_steps=5,
    max_iterations=200,
):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644

    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.

    :param model_fn: a callable that takes an input tensor and returns
              the model logits. The logits should be a tensor of shape
              (n_examples, n_classes).
    :param x: input tensor of shape (n_examples, ...), where ... can
              be any arbitrary dimension that is compatible with
              model_fn.
    :param n_classes: the number of classes.
    :param y: (optional) Tensor with true labels. If targeted is true,
              then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when
              crafting adversarial samples. Otherwise, model predictions
              are used as labels to avoid the "label leaking" effect
              (explained in this paper:
              https://arxiv.org/abs/1611.01236). If provide y, it
              should be a 1D tensor of shape (n_examples, ).
              Default is None.
    :param targeted: (optional) bool. Is the attack targeted or
              untargeted? Untargeted, the default, will try to make the
              label incorrect. Targeted will instead try to move in the
              direction of being more like y.
    :param lr: (optional) float. The learning rate for the attack
              algorithm. Default is 5e-3.
    :param confidence: (optional) float. Confidence of adversarial
              examples: higher produces examples with larger l2
              distortion, but more strongly classified as adversarial.
              Default is 0.
    :param clip_min: (optional) float. Minimum float value for
              adversarial example components. Default is 0.
    :param clip_max: (optional) float. Maximum float value for
              adversarial example components. Default is 1.
    :param initial_const: The initial tradeoff-constant to use to tune the
              relative importance of size of the perturbation and
              confidence of classification. If binary_search_steps is
              large, the initial constant is not important. A smaller
              value of this constant gives lower distortion results.
              Default is 1e-2.
    :param binary_search_steps: (optional) int. The number of times we
              perform binary search to find the optimal tradeoff-constant
              between norm of the perturbation and confidence of the
              classification. Default is 5.
    :param max_iterations: (optional) int. The maximum number of
              iterations. Setting this to a larger value will produce
              lower distortion results. Using only a few iterations
              requires a larger learning rate, and will produce larger
              distortion results. Default is 1000.
    """

    # def compare(pred, label, is_logits=False):
    #     """
    #     A helper function to compare prediction against a label.
    #     Returns true if the attack is considered successful.
    #
    #     :param pred: can be either a 1D tensor of logits or a predicted
    #             class (int).
    #     :param label: int. A label to compare against.
    #     :param is_logits: (optional) bool. If True, treat pred as an
    #             array of logits. Default is False.
    #     """
    #
    #     # Convert logits to predicted class if necessary
    #     if is_logits:
    #         pred_copy = pred.clone().detach()
    #         pred_copy[label] += -confidence if targeted else confidence
    #         pred = torch.argmax(pred_copy)
    #
    #     return pred == label if targeted else pred != label

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos1 = nn.CosineSimilarity(dim=2, eps=1e-6)

    # if y is None:
    #     # Using model predictions as ground truth to avoid label leaking
    #     pred = model_fn(x)
    #     y = torch.argmax(pred, 1)

    # Initialize some values needed for binary search on const
    lower_bound = [0.0] * len(x)
    upper_bound = [1e10] * len(x)
    const = x.new_ones(len(x), 1) * initial_const
    # print(initial_const,const,x.shape,len(x))
    # exit()


    o_bestl2 = [INF] * len(x)
    o_bestscore = [-1.0] * len(x)

    # x = torch.clamp(x, clip_min, clip_max)
    ox = ori_x.clone().detach()  # save the original x
    o_bestattack = x.clone().detach()

    # Map images into the tanh-space
    # x = (x - clip_min) / (clip_max - clip_min)

    # x = torch.clamp(x, 0, 1)
    # x = x * 2 - 1 #(-1,1)
    # print(torch.max(x),torch.min(x))
    x = torch.arctanh(x * 0.999999)
    # print(torch.max(x), torch.min(x))
    # exit()

    # Prepare some variables
    modifier = torch.zeros_like(x, requires_grad=True)
    # y_onehot = torch.nn.functional.one_hot(y, n_classes).to(torch.float)

    # Define loss functions and optimizer
    # f_fn = lambda real, other, targeted: torch.max(
    #     ((other - real) if targeted else (real - other)) + confidence,
    #     torch.tensor(0.0).to(real.device),
    # )
    # print(list(range(len(x.size())))[1:])
    # exit()
    l2dist_fn = lambda x, y: torch.pow(x - y, 2).sum(list(range(len(x.size())))[1:])
    optimizer = torch.optim.Adam([modifier], lr=lr)
    succeeded=0
    succeeded_past=0

    # Outer loop performing binary search on const
    for outer_step in range(binary_search_steps):
        # Initialize some values needed for the inner loop
        # bestl2 = [INF] * len(x)
        # bestscore = [-1.0] * len(x)

        # Inner loop performing attack iterations
        for i in range(max_iterations):
            # One attack step
            new_x = torch.tanh(modifier + x) #+ 1) / 2
            # new_x = new_x * (clip_max - clip_min) + clip_min
            # print('max,min',torch.max(new_x),torch.min(new_x))

            out = model_fn(new_x)
            feat_len = min(out[2].shape[1], y[2].shape[1])
            out[2] = out[2][:, :feat_len, :]
            y[2] = y[2][:, :feat_len, :]
            num_feats = out[1].shape[0] + out[2].shape[0] * out[2].shape[1]
            f=torch.sum((-cos(out[1], y[1]) + torch.sum(-cos1(out[2], y[2]), 1)))/num_feats
            optimizer.zero_grad()
            # f = f_fn(real, other, targeted)
            l2 = l2dist_fn(new_x, ox)
            loss = (const * f + l2).sum()
            loss.backward(retain_graph=True)
            # loss.requires_grad_(True).backward()
            optimizer.step()
            if i%10==0:
                for n, (l2_n, new_x_n) in enumerate(zip(l2, new_x)):
                    succeeded = model_fn_compare(new_x_n)

                    if l2_n < o_bestl2[n] and succeeded:
                        succeeded_past = 1
                        o_bestl2[n] = l2_n
                        o_bestattack[n] = new_x_n
                        return o_bestattack.detach(),succeeded,o_bestl2
        for n, (l2_n, new_x_n) in enumerate(zip(l2, new_x)):
            succeeded = model_fn_compare(new_x_n)

            if l2_n < o_bestl2[n] and succeeded:
                succeeded_past = 1
                o_bestl2[n] = l2_n
                #         # o_bestscore[n] = pred_n
                o_bestattack[n] = new_x_n
                return o_bestattack.detach(), succeeded, o_bestl2
            else:
                lower_bound[n] = max(lower_bound[n], const[n])
                if upper_bound[n] < 1e9:
                    const[n] = (lower_bound[n] + upper_bound[n]) / 2
                else:
                    const[n] *= 10
    if succeeded_past==0:
        # print(2222)
        o_bestattack=new_x
        # print(torch.max(o_bestattack-ori_x),torch.min(o_bestattack-ori_x))

    #up 1e10 1e10 1         1
    #low 0  0.1  0.1         0.55
    #c   0.1 1    (1+0.1)/2  (1+0.55)/2
    return o_bestattack.detach(),succeeded,o_bestl2


if __name__ == "__main__":
    x = torch.clamp(torch.randn(5, 10), 0, 1)
    y = torch.randint(0, 9, (5,))
    model_fn = lambda x: x

    # targeted
    new_x = carlini_wagner_l2(model_fn, x, 10, targeted=True, y=y)
    new_pred = model_fn(new_x)
    new_pred = torch.argmax(new_pred, 1)

    # untargeted
    new_x_untargeted = carlini_wagner_l2(model_fn, x, 10, targeted=False, y=y)
    new_pred_untargeted = model_fn(new_x_untargeted)
    new_pred_untargeted = torch.argmax(new_pred_untargeted, 1)
