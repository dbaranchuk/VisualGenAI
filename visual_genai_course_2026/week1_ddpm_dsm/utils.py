import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def show_image_batch(
    batch: np.ndarray,
    n_cols: int = 8,
    figsize_per_cell: float = 2.5,
    normalize: bool = False,
    titles=None,
):
    if batch.ndim == 3:  # [B,H,W] -> [B,H,W,1]
        batch = batch[..., None]

    b, h, w, c = batch.shape
    n_cols = max(1, int(n_cols))
    n_rows = math.ceil(b / n_cols)

    fig_w = n_cols * figsize_per_cell
    fig_h = n_rows * figsize_per_cell
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= b:
            continue

        img = batch[i]

        if normalize:
            img = img.astype(np.float32)
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) / (mx - mn)

        if c == 1:
            ax.imshow(
                img[..., 0],
                cmap="gray",
                vmin=0,
                vmax=1 if normalize else None,
            )
        else:
            ax.imshow(img)

        if titles is not None and i < len(titles):
            ax.set_title(str(titles[i]), fontsize=10)

    plt.tight_layout()
    plt.show()
    return fig


# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    extra_tokens: int = 0,
):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or
               [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed],
            axis=0,
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray):
    assert embed_dim % 2 == 0

    # Use half of dimensions to encode grid_h.
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
        )
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift, scale = self.ada_ln_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def make_respaced_betas(alphas_cumprod: torch.Tensor, use_timesteps):
    """
    use_timesteps: ascending list/array of original timesteps (len = M)
    """
    use_timesteps = list(map(int, use_timesteps))
    last_alpha_bar = 1.0
    new_betas = []

    for t in use_timesteps:
        alpha_bar_t = float(alphas_cumprod[t].item())
        new_beta = 1.0 - alpha_bar_t / last_alpha_bar
        new_betas.append(new_beta)
        last_alpha_bar = alpha_bar_t

    return (
        np.array(new_betas, dtype=np.float64),
        np.array(use_timesteps, dtype=np.int64),
    )


def _extract_into_tensor(
    arr: torch.Tensor,
    timesteps: torch.Tensor,
    broadcast_shape,
):
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    alpha_bar,
    max_beta: float = 0.999,
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0, 1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
        produces the cumulative product of (1-beta) up to that part of the
        diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
        prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)