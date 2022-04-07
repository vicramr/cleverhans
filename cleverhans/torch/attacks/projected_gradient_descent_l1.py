"""
Modification of PGD to add a regularization term to the loss.
"""
import numpy as np
import torch

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.utils import clip_eta
from cleverhans.torch.utils import optimize_linear


def projected_gradient_descent_l1(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    rand_init=True,
    rand_minmax=None,
    sanity_checks=True,
    l1_penalty=1.0,
    iters_before_l1=None,
):
    """
    Parameters present in projected_gradient_descent are the same as in that function.
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
    :param l1_penalty: float. The coefficient for the L1 penalty term.
    :param iters_before_l1: (optional) int. If given, this many iterations of vanilla PGD will
              be done before the L1 regularization term is added. This does not change the total
              number of iterations which are done.
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
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
    else:
        eta = torch.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    if clip_min is not None or clip_max is not None:
        # clip eta again such that clip_min <= adv_x <= clip_max
        # Needed because we want eta to correctly capture all clipping, as
        # we are applying regularization directly to eta
        eta = torch.clamp(eta, clip_min - x, clip_max - x)
    adv_x = x + eta

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    i = 0
    while i < nb_iter:
        adv_x = adv_x.clone().detach().to(torch.float).requires_grad_(True)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(model_fn(adv_x), y)
        # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
        if targeted:
            loss = -loss
        if iters_before_l1 is not None and iters_before_l1 <= i:
            l1_loss = l1_penalty * torch.sum(torch.abs(eta))
            combined_loss = loss - l1_loss # Subtract l1_loss because combined_loss is being maximized
        else:
            combined_loss = loss

        combined_loss.backward()
        curr_perturbation = optimize_linear(adv_x.grad, eps_iter, norm) # FGM step, valid for both L-infinity and L2
        eta += curr_perturbation # eta accumulates the total perturbation so far
        # Clip eta such that adv_x is in the norm ball
        eta = clip_eta(eta, norm, eps)
        # Also clip such that adv_x elements are within [clip_min, clip_max]
        if clip_min is not None or clip_max is not None:
            eta = torch.clamp(eta, clip_min - x, clip_max - x)
        adv_x = x + eta
        i += 1

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast clip_min and clip_max to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x
