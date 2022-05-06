"""
Black-box projected gradient descent attack, using Natural Evolutionary Strategies (NES) to
turn white-box PGD into a black-box attack. Each backward pass is approximated using multiple
forward passes.
Paper link: https://proceedings.mlr.press/v80/ilyas18a.html
"""
import numpy as np
import torch

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.utils import clip_eta, optimize_linear, nes


def projected_gradient_descent_nes(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    n=50,
    sigma=0.001,
    momentum=None,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    rand_init=True,
    rand_minmax=None,
    sanity_checks=True,
):
    """
    Parameters present in projected_gradient_descent are the same as in that function.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param n: (optional) positive int. Half the number of queries per iteration for gradient
              estimation.
    :param sigma: (optional) positive float. The search variance. See the NES paper (linked
              above) for details. This is equivalent to sigma in that paper.
    :param momentum: (optional) positive float. This is a momentum value used when performing
              gradient descent. Should be in the range (0, 1). The original method's code
              uses a default value of 0.9.
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
    
    if x.shape[0] != 1:
        raise ValueError(
            "This function currently only supports tensors with a batch size of 1"
        )
    
    if n <= 0:
        raise ValueError(
            "n must be positive"
        )
    
    if sigma <= 0:
        raise ValueError(
            "sigma must be positive"
        )
    
    if momentum is not None and (momentum < 0 or momentum > 1):
        raise ValueError(
            "If given, momentum should be in range (0, 1)"
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
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    i = 0
    while i < nb_iter:
        adv_x = adv_x.clone().detach().to(torch.float).requires_grad_(True)
        loss_fn_unreduced = torch.nn.CrossEntropyLoss(reduction='none')
        if targeted:
            # negate the output of loss_fn so that we minimize the loss of the target label
            loss_fn = lambda output, label: -loss_fn_unreduced(output, label)
        else:
            loss_fn = loss_fn_unreduced

        grad_estimate = nes(model_fn, adv_x, y, loss_fn, n, sigma)
        if (momentum is not None) and (i > 0):
            grad_estimate = (momentum * prev_grad) + ((1.0 - momentum) * grad_estimate)

        optimal_perturbation = optimize_linear(grad_estimate, eps_iter, norm)

        adv_x = adv_x + optimal_perturbation

        # Clipping perturbation eta to norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Set up for next iteration's momentum computation
        prev_grad = grad_estimate.detach() # detaching helps prevent memory overflow

        # Also clip to given bounds
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
        i += 1
# TODO check order of clipping. Clip again like fast_gradient_method would?
    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast clip_min and clip_max to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x
