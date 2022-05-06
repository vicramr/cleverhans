"""
TODO
This only works for convolutional neural networks where the inputs have shape (batch_size, n_channels, *, *)
"""
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.utils import clip_eta
from cleverhans.torch.utils import optimize_linear


def projected_gradient_descent_gradcam(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    gradcam_target_layers,
    gradcam_threshold,
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
    :param gradcam_target_layers: the list of layers which will be used when initializing
              GradCAM.
    :param gradcam_threshold: float in range [0,1]. Any pixels with grad-cam value less than
              this will not have their value changed in the adversarial example.
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
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
    else:
        eta = torch.zeros_like(x)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    # Compute grad-cam for x to get mask
    with GradCAM(model=model_fn, target_layers=gradcam_target_layers, use_cuda=torch.cuda.is_available()) as cam:
        y_as_list = [y[i].item() for i in range(y.shape[0])] # TODO is this sufficient for the attack to work for targeted in addition to untargeted attacks?
        gradcam_targets = [ClassifierOutputTarget(y_i) for y_i in y_as_list]
        cam_output = cam(input_tensor=x, targets=gradcam_targets)
        assert cam_output.shape == (x.shape[0], x.shape[2], x.shape[3])
        cam_output_expanded = cam_output.reshape(x.shape[0], 1, x.shape[2], x.shape[3])
        cam_output_masked = torch.from_numpy(cam_output_expanded >= gradcam_threshold).to(torch.float).to(x.device)
        mask = torch.cat([cam_output_masked for _ in range(x.shape[1])], dim=1) # Stack a copy of mask for each channel (could also be achieved using broadcasting)
        assert mask.shape == x.shape

    # Clip eta and zero out elements below the grad-cam threshold
    eta = clip_eta(eta, norm, eps)
    if clip_min is not None or clip_max is not None:
        # clip eta again such that clip_min <= adv_x <= clip_max
        # Needed because we want eta to correctly capture all clipping, as
        # we are applying regularization directly to eta
        eta = torch.clamp(eta, clip_min - x, clip_max - x)
    eta = torch.mul(eta, mask)
    adv_x = x + eta

    i = 0
    while i < nb_iter:
        adv_x = adv_x.clone().detach().to(torch.float).requires_grad_(True)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(model_fn(adv_x), y)
        # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
        if targeted:
            loss = -loss

        loss.backward()
        curr_perturbation = optimize_linear(adv_x.grad, eps_iter, norm) # FGM step, valid for both L-infinity and L2
        eta += curr_perturbation # eta accumulates the total perturbation so far
        # Clip eta such that adv_x is in the norm ball
        eta = clip_eta(eta, norm, eps)
        # Also clip such that adv_x elements are within [clip_min, clip_max]
        if clip_min is not None or clip_max is not None:
            eta = torch.clamp(eta, clip_min - x, clip_max - x)
        # Zero out elements below gradcam threshold
        eta = torch.mul(eta, mask)
        adv_x = x + eta
        i += 1

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast clip_min and clip_max to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x
