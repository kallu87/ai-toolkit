import math

import torch


def signed_advantages(scores: list[int], samples_per_prompt: int) -> torch.Tensor:
    groups = torch.tensor(scores, dtype=torch.float32).reshape(-1, samples_per_prompt)
    centered = groups - groups.mean(dim=1, keepdim=True)
    return (centered / (groups.std(dim=1, correction=0, keepdim=True) + 1e-6)).flatten()


def flow_batch(clean: torch.Tensor, generator: torch.Generator, device: torch.device):
    # Independent Gaussian noise for forward regression. MOSS runs with sigma=1
    # at the noise endpoint, so the velocity target has the opposite sign to Eq. 4.
    noise = torch.randn(clean.shape, dtype=torch.float32, generator=generator).to(device)
    sigmas = torch.rand(clean.shape[0], dtype=torch.float32, generator=generator).to(device)
    clean = clean.to(device, dtype=torch.float32)
    noisy = torch.lerp(clean, noise, sigmas.reshape(-1, 1, 1, 1))
    return noisy, sigmas * 1000, noise - clean


def velocity_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.float() - target.float()).square().flatten(1).mean(1)


def update_multiplier(multiplier, drift_ema, budget, rate, minimum, maximum):
    error = max(-1.0, min(1.0, drift_ema / budget - 1.0))
    log_multiplier = math.log(multiplier) + rate * error
    return math.exp(max(math.log(minimum), min(math.log(maximum), log_multiplier)))
