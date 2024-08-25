import torch
import einops
import einops_exts
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
from diffusers.models.embeddings import PatchEmbed

def get_conditioner(
        con_type,
        con_dim,
        con_depth,
        con_numq,
        con_nframes,
        con_patch_size,
        con_dim_head,
        con_heads,
):
    if con_type == "qformer":
        conditioner = Noisy_Conditioner_Q_Former(
            dim = con_dim,
            depth = con_depth,
            num_q = con_numq,
            num_frames = con_nframes,
            patch_size = con_patch_size,
            dim_head = con_dim_head,
            heads = con_heads,
        )
    else:
        raise ValueError(f"Unknown conditioner type: {con_type}")

    return conditioner


class Noisy_Conditioner_Q_Former(nn.Module):
    def __init__(
            self,
            dim        = 1024,
            depth      = 2,
            num_q      = 512,
            num_frames = 64,
            patch_size = 8,
            dim_head   = 88,
            heads      = 16,
            ff_mult    = 4,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbed(
            height=64,
            width=64,
            patch_size=patch_size,
            in_channels=4,
            embed_dim=dim,
            interpolation_scale=1,
        )
        self.latents = nn.Parameter(torch.randn(num_q, dim))
        self.frame_pos_emb = nn.Parameter(torch.randn(num_frames, dim))
        self.layers = nn.ModuleList([])
        self.frame_emb = nn.Embedding(num_frames, dim)
        self.context_time_emb = nn.Embedding(1000, dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult)
            ]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, timesteps, random_frame_indices):
        b, f, c, h, w = x.shape
        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = rearrange(self.patch_embedding(x), "(b f) num_patch d -> b f num_patch d", f=f)

        # if random_frame_indices is None:
        #     random_frame_indices = torch.arange(f).to(x.device)
        #     x = x.repeat(f, 1, 1, 1)
            # timesteps = timesteps.repeat(f)

        time_token = self.context_time_emb(timesteps)[:, None, :]
        frame_token = self.frame_emb(random_frame_indices).unsqueeze(dim=1)

        x = x + self.frame_pos_emb[:f].unsqueeze(dim=1)
        x = rearrange(x, "b f num_patch d -> b (f num_patch) d")

        x = torch.cat((time_token, frame_token, x), dim=-2)

        latents = repeat(self.latents, "num_q d -> b num_q d", b=x.shape[0])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        result = self.norm(latents)

        return result


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, latents):
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads
        q = self.to_q(latents)

        # x: (b, n, d), latents: (b, num_latents, d), kv_input: (b, n+num_latents, d)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1) # (b, n+num_latents, inner_dim * 2)

        q, k, v = einops_exts.rearrange_many((q, k, v), "b n (h d) -> b h n d", h = h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k) # i = num_latents of q, j = n + num_latents
        # if indices is not None:
        #     mask = torch.zeros_like(sim) # (b, h, i, j)
        #     b, n, d = x.shape
        #     slice_size = n // f
        #     for i, idx in enumerate(indices):
        #         start_idx = idx * slice_size
        #         mask[i, :, :, start_idx:(start_idx+slice_size)] = -torch.inf
        #     sim = sim + mask
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)", h = h)

        return self.to_out(out)


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )