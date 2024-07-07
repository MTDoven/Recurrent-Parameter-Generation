import torch
import torch.nn as nn
import torch.nn.functional as F
from .denoiser import ConditionalMLP
from tqdm import tqdm
import math


class GaussianDiffusion(nn.Module):
    def __init__(self, device, beta_max=0.999, n_timesteps=1000):
        super().__init__()
        self.device = device
        self.n_timesteps = n_timesteps
        # Define beta schedule
        def betas_for_alpha_bar(n_timesteps, max_beta=0.999):
            gen = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = [min(1 - gen((i + 1) / n_timesteps) / gen(i / n_timesteps), max_beta)
                     for i in range(n_timesteps)]
            return torch.tensor(betas)

        # Define beta schedule
        betas = betas_for_alpha_bar(n_timesteps, beta_max).to(device)
        alphas = 1. - betas
        sqrt_betas = torch.sqrt(betas)
        one_divide_sqrt_alphas = 1 / torch.sqrt(alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        betas_divide_sqrt_one_minus_alphas_cumprod = betas / sqrt_one_minus_alphas_cumprod
        # register buffer
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('sqrt_betas',
                             sqrt_betas)
        self.register_buffer('one_divide_sqrt_alphas',
                             one_divide_sqrt_alphas)
        self.register_buffer('alphas_cumprod',
                             alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod',
                             sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             sqrt_one_minus_alphas_cumprod)
        self.register_buffer('betas_divide_sqrt_one_minus_alphas_cumprod',
                             betas_divide_sqrt_one_minus_alphas_cumprod)

    def q_sample(self, x, t, noise):
        # Diffusion forward process: q(x_t | x_0).
        t = t.unsqueeze(-1).unsqueeze(-1)
        return self.sqrt_alphas_cumprod[t] * x + \
               self.sqrt_one_minus_alphas_cumprod[t] * noise

    def p_sample(self, x, t, z, model):
        # Diffusion reverse process: p_theta(x_{t-1} | x_t).
        pred_noise = model(x, t, z)
        mu = self.one_divide_sqrt_alphas[t] * \
              (x - self.betas_divide_sqrt_one_minus_alphas_cumprod[t] * pred_noise)
        return mu if t == 0 else mu + self.sqrt_betas[t] * torch.randn_like(x)

    def p_sample_ddim(self, x, t, z, model, sample_timesteps=100, eta=0.05, t_next=None):
        # Diffusion reverse process: p_theta(x_{t-tao} | x_t).
        if t == 0:
            return self.p_sample(x, t, z, model)
        t_next = max(sample_timesteps - self.n_timesteps // sample_timesteps if t_next is None else t_next, 0)
        sigma = eta * ((1. - self.alphas_cumprod[t] / self.alphas_cumprod[t_next]) *
                       (1. - self.alphas_cumprod[t_next]) / (1. - self.alphas_cumprod[t])).sqrt()
        pred_noise = model(x, t, z)
        pred_x0 = (x - self.sqrt_one_minus_alphas_cumprod[t] * pred_noise) / self.sqrt_alphas_cumprod[t]
        pred_xt = (1. - self.alphas_cumprod[t] - sigma ** 2.).sqrt() * pred_noise
        x = self.sqrt_alphas_cumprod[t_next] * pred_x0 + pred_xt + sigma * torch.randn_like(x)
        return x


class DiffusionLoss(nn.Module):
    def __init__(self, mlp_layer_dims: list, condition_dim: int,
                 diffusion_beta_max=0.999, diffusion_n_timesteps=1000,
                 device=torch.device("cpu"), mlp_activation=nn.SiLU(),):
        super().__init__()
        self.device = device
        self.net = ConditionalMLP(layer_dims=mlp_layer_dims,
                                  condition_dim=condition_dim,
                                  device=device,
                                  activation=mlp_activation)
        self.diffusion = GaussianDiffusion(device=device,
                                           beta_max=diffusion_beta_max,
                                           n_timesteps=diffusion_n_timesteps)

    def forward(self, x, z, **kwargs):
        if self.training:
            return self.loss(x, z)
        else:  # evaluate
            return self.sample_ddim(x, z, **kwargs)

    def loss(self, x, z):
        # Given condition z and ground truth token x, compute loss
        t = torch.randint(0, self.diffusion.n_timesteps, (x.size(0),), device=self.device)
        noise_origin = torch.randn(x.shape, device=self.device)
        x_t = self.diffusion.q_sample(x, t, noise_origin)
        noise_pred = self.net(x_t, t, z)
        loss = F.mse_loss(noise_pred, noise_origin)
        return loss

    def sample(self, x, z, **kwargs):
        # Given condition and noise, sample x using reverse diffusion process
        for t in tqdm(list(range(self.diffusion.n_timesteps))[::-1]):
            x = self.diffusion.p_sample(x, torch.tensor([t], device=self.device), z, self.net)
        return x

    def sample_ddim(self, x, z, sample_timesteps=100, eta=0.05):
        # Given condition and noise, sample x using reverse diffusion process
        times = torch.linspace(-1, self.diffusion.n_timesteps-1, steps=sample_timesteps+1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        for t, t_next in tqdm(time_pairs):
            x = self.diffusion.p_sample_ddim(
                    x, torch.tensor([t], device=self.device), z,
                    self.net, sample_timesteps, eta, t_next)
        return x


if __name__ == '__main__':
    model = DiffusionLoss(mlp_layer_dims=[2048 for i in range(3)], condition_dim=4096,
                          device=torch.device('cuda'), mlp_activation=nn.SiLU()).to("cuda")
    x = torch.randn(4, 32, 2048).to("cuda")
    z = torch.randn(4, 32, 4096).to("cuda")
    loss = model.loss(x, z)
    print(loss)
    y = model.sample(x, z)
    print(y.shape)
    y = model.sample_ddim(x, z, sample_timesteps=50, eta=0.05)
    print(y.shape)