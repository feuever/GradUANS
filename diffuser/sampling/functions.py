import torch

from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)
from test_ensemble import load_ensemble_models , compute_uncertainty 

@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
     uncertainty_scale=0.1  
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    if scale != 0:
        for _ in range(n_guide_steps):
            with torch.enable_grad():
                y, grad = guide.gradients(x, cond, t)
                uncertainty_grad = compute_uncertainty_gradient(x)
                grad = grad - uncertainty_scale * uncertainty_grad

            if scale_grad_by_std:
                grad = model_var * grad

            grad[t < t_stopgrad] = 0

            x = x + scale * grad
            x = apply_conditioning(x, cond, model.action_dim, model.goal_dim, k=t[0])

    # 计算不确定度并将其作为'un'参数传递
    uncertainty = compute_uncertainty(x)
    # 将标量不确定度转换为与x形状匹配的张量
    un = torch.ones_like(x[..., 0]) * uncertainty
    # 然后传递给p_mean_variance
    model_mean, _, model_log_variance = model.p_mean_variance(x, cond, t, un=un)
    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y, uncertainty  # 返回不确定度


def compute_uncertainty_gradient(x):

    x_with_grad = x.detach().requires_grad_(True)
    # 调用test_ensemble中的compute_uncertainty方法
    uncertainty = compute_uncertainty(x_with_grad)
    uncertainty_grad = torch.autograd.grad(uncertainty.sum(), x_with_grad)[0]
    return -uncertainty_grad  # 取负号表示向不确定度减小的方向