"""
The SLIDE attack with a variation: instead of using the gradient itself to
determine which pixels to modify in each iteration, we use grad-CAM.
This only works for convolutional neural networks where the inputs have shape (batch_size, n_channels, *, *)
"""
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from cleverhans.torch.utils import zero_out_clipped_grads


def sparse_l1_descent_gradcam(
    model_fn,
    x,
    gradcam_target_layers,
    eps=10.0,
    eps_iter=1.0,
    nb_iter=20,
    y=None,
    targeted=False,
    clip_min=None,
    clip_max=None,
    rand_init=False,
    clip_grad=False,
    grad_sparsity=99,
    sanity_checks=True,
):
    """
    This class implements a variant of Projected Gradient Descent for the l1-norm
    (Tramer and Boneh 2019). The l1-norm case is more tricky than the l-inf and l2
    cases covered by the ProjectedGradientDescent class, because the steepest
    descent direction for the l1-norm is too sparse (it updates a single
    coordinate in the adversarial perturbation in each step). This attack has an
    additional parameter that controls the sparsity of the update step. For
    moderately sparse update steps, the attack vastly outperforms Projected
    Steepest Descent and is competitive with other attacks targeted at the l1-norm
    such as the ElasticNetMethod attack (which is much more computationally
    expensive).
    Paper link (Tramer and Boneh 2019): https://arxiv.org/pdf/1904.13000.pdf

    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param gradcam_target_layers: the list of layers which will be used when initializing
        GradCAM.
    :param eps: (optional float) maximum distortion of adversarial example
        compared to original input
    :param eps_iter: (optional float) step size for each attack iteration
    :param nb_iter: (optional int) Number of attack iterations.
    :param y: (optional) A tensor with the true labels.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
        Untargeted, the default, will try to make the label incorrect.
        Targeted will instead try to move in the direction of being more like y.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    :param clip_grad: (optional bool) Ignore gradient components
        at positions where the input is already at the boundary
        of the domain, and the update step will get clipped out.
    :param grad_sparsity (optional) Relative sparsity of the gradient update
        step, in percent. This parameter should be a scalar.
        Example: if grad_sparsity is 70, then the 70% of the pixels with the
        lowest grad-CAM values will not be updated, and the remaining 30% of
        pixels will be updated.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
        memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial examples
    """
    if clip_grad and (clip_min is None or clip_max is None):
        raise ValueError("Must set clip_min and clip_max if clip_grad is set")

    # The grad_sparsity argument governs the sparsity of the gradient
    # update. It indicates the percentile value above which gradient entries
    # are retained. It can be specified as a scalar or as a 1-dimensional
    # tensor of the same size as the input's batch dimension.
    if isinstance(grad_sparsity, int) or isinstance(grad_sparsity, float):
        if not 0 < grad_sparsity < 100:
            raise ValueError("grad_sparsity should be in (0, 100)")
    else:
        raise ValueError("grad_sparsity should be an int or float")

    asserts = []

    # eps_iter should be at most eps
    asserts.append(eps_iter <= eps)

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

    if sanity_checks:
        assert np.all(asserts)

    # Initialize loop variables
    if rand_init:
        dist = torch.distributions.laplace.Laplace(
            torch.tensor([1.0]), torch.tensor([1.0])
        )
        dim = torch.prod(torch.tensor(x.shape[1:]))
        eta = dist.sample([x.shape[0], dim]).squeeze(-1).to(x.device)
        norm = torch.sum(torch.abs(eta), axis=-1, keepdim=True)
        w = torch.pow(
            torch.rand(x.shape[0], 1, device=x.device), torch.tensor(1.0 / dim)
        )
        eta = torch.reshape(eps * (w * eta / norm), x.shape)
    else:
        eta = torch.zeros_like(x)

    # Clip eta
    adv_x = x.clone().detach().requires_grad_(True)
    eta = eta.renorm(p=1, dim=0, maxnorm=eps)
    adv_x = adv_x + eta

    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(x, clip_min, clip_max)

    if y is None:
        y = torch.argmax(model_fn(x), 1)

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    for i in range(nb_iter):
        adv_x = adv_x.clone().detach().to(torch.float).requires_grad_(True)
        logits = model_fn(adv_x)

        # Compute loss
        loss = criterion(logits, y)
        if targeted:
            loss = -loss

        # Define gradient of loss wrt input
        (grad,) = torch.autograd.grad(loss.mean(), [adv_x])

        if clip_grad:
            grad = zero_out_clipped_grads(grad, adv_x, clip_min, clip_max)

        grad_view = grad.view(grad.shape[0], -1)

        # Compute grad-CAM for masking
        with GradCAM(model=model_fn, target_layers=gradcam_target_layers, use_cuda=torch.cuda.is_available()) as cam:
            y_as_list = [y[i].item() for i in range(y.shape[0])] # TODO is this sufficient for the attack to work for targeted in addition to untargeted attacks?
            gradcam_targets = [ClassifierOutputTarget(y_i) for y_i in y_as_list]
            cam_output = cam(input_tensor=x, targets=gradcam_targets)
            assert cam_output.shape == (x.shape[0], x.shape[2], x.shape[3])
            cam_output = torch.from_numpy(cam_output).to(x.device)
        # Use cam_output to get the top pixels
        cam_output_view = cam_output.view(cam_output.shape[0], -1)
        k = int(grad_sparsity / 100.0 * cam_output_view.shape[1])
        percentile_value, _ = torch.kthvalue(cam_output_view, k, keepdim=True)
        assert percentile_value.shape == (cam_output.shape[0], 1)
        # Stack cam_output n_channels times, then re-flatten it so we can
        # repeat percentile_value based on its new shape. Then use this to
        # compute tied_for_max and num_ties in the way that the original
        # sparse_l1_descent attack does.
        cam_output_expanded = torch.reshape(cam_output, (x.shape[0], 1, x.shape[2], x.shape[3]))
        cam_output_stacked = torch.cat([cam_output_expanded for _ in range(x.shape[1])], dim=1) # Stack a copy of cam_output_expanded per channel (similar to broadcasting)
        assert cam_output_stacked.shape == x.shape
        cam_output_stacked_view = cam_output_stacked.view(cam_output_stacked.shape[0], -1)
        percentile_value_repeated = percentile_value.repeat(1, cam_output_stacked_view.shape[1])
        assert percentile_value_repeated.shape == cam_output_stacked_view.shape == grad_view.shape
        tied_for_max = torch.ge(cam_output_stacked_view, percentile_value_repeated).int().float()
        num_ties = torch.sum(tied_for_max, dim=1, keepdim=True)

        # With tied_for_max and num_ties, compute optimal_perturbation in the normal way
        optimal_perturbation = (torch.sign(grad_view) * tied_for_max) / num_ties
        optimal_perturbation = optimal_perturbation.view(grad.shape)

        # Add perturbation to original example to obtain adversarial example
        adv_x = adv_x + optimal_perturbation * eps_iter

        # If clipping is needed, reset all values outside of [clip_min, clip_max]
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)

        # Clipping perturbation eta to the l1-ball
        eta = adv_x - x
        eta = eta.renorm(p=1, dim=0, maxnorm=eps)
        adv_x = x + eta

        # Redo the clipping.
        # Subtracting and re-adding eta can add some small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)

    return adv_x.detach()
