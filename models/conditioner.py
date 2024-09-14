import torch
import einops
import einops_exts
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
from diffusers.models.embeddings import PatchEmbed
from diffusers.models.attention import Attention

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


class Context_Transformer(nn.Module):
    def __init__(
        self,
        dim,    
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn_spatial = Attention(
            dim,
            heads=16,
            dim_head=88,
        )
        self.attn_temporal = Attention(
            dim,
            heads=16,
            dim_head=88,
        )
        self.ff = FeedForward(
            dim=dim,
            mult=4
        )
        
    def forward(self, x):
        """
        x: (b, f, l, d)
        """
        b, f, l, d = x.shape
        x_spatial = rearrange(x, "b f l d -> (b f) l d").contiguous()
        x_spatial = self.norm1(x_spatial)
        x_spatial = self.attn_spatial(x_spatial)
        x_spatial = rearrange(x_spatial, "(b f) l d -> b f l d", f=f)
        x = x + x_spatial

        x_temporal = rearrange(x, "b f l d -> (b l) f d").contiguous()
        x_temporal = self.attn_temporal(x_temporal)
        x = x + rearrange(x_temporal, "(b l) f d -> b f l d", l=l)
        x = self.norm2(x)
        x = self.ff(x)

        return x


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

    def forward(self, x, latents, show=False):
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads
        q = self.to_q(latents)

        # x: (b, n, d), latents: (b, num_latents, d), kv_input: (b, n+num_latents, d)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1) # (b, n+num_latents, inner_dim * 2)
        # k, v = self.to_kv(x).chunk(2, dim=-1)

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
        # if show:
        #     import matplotlib.pyplot as plt
        #     plt.matshow(attn[0].mean(dim=0).cpu().numpy())
        #     plt.colorbar()
        #     plt.savefig("samples/attn.png")
        #     plt.close()

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

class Window_Layer(nn.Module):
    def __init__(self, dim, dim_head, heads, ff_mult):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            heads=heads,
            dim_head=dim_head,
        )
        self.ff = FeedForward(dim, ff_mult)

    def forward(self, x):
        x = self.norm(x)
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class Window_Transformer(nn.Module):
    def __init__(self, dim, dim_head, heads, ff_mult, layers):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(Window_Layer(dim, dim_head, heads, ff_mult))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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
        # self.st_transformer = Context_Transformer(dim)
        # self.window_transformer = Window_Transformer(dim, dim_head, heads, ff_mult, 4)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # nn.LayerNorm(dim),
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
                # nn.LayerNorm(dim),
                # Attention(dim, heads=heads, dim_head=dim_head),
                # FeedForward(dim=dim, mult=ff_mult),
            ]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, timesteps, random_frame_indices):
        b, f, c, h, w = x.shape
        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = rearrange(self.patch_embedding(x), "(b f) num_patch d -> b f num_patch d", f=f)

        time_token = self.context_time_emb(timesteps)[:, None, :]
        frame_token = self.frame_emb(random_frame_indices).unsqueeze(dim=1)

        x = x + self.frame_pos_emb[:f].unsqueeze(dim=1)
        x = rearrange(x, "b f num_patch d -> b (f num_patch) d")

        x = torch.cat((time_token, frame_token, x), dim=-2)

        latents = repeat(self.latents, "num_q d -> b num_q d", b=x.shape[0])

        for i, (attn, ff) in enumerate(self.layers):
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        # for i, (cross_norm, cross_attn, cross_ff, self_norm, self_attn, self_ff) in enumerate(self.layers):
        #     latents = cross_norm(latents)
        #     latents = cross_attn(x, latents) + latents
        #     latents = cross_ff(latents) + latents
            
        #     latents = self_norm(latents)
        #     latents = self_attn(latents) + latents
        #     latents = self_ff(latents) + latents

        result = self.norm(latents)

        return result

if __name__ == "__main__":
    device = torch.device("cuda:7")
    conditioner = get_conditioner(
        con_type = "qformer",
        con_dim = 1024,
        con_depth = 20,
        con_numq = 1024,
        con_nframes = 64,
        con_patch_size = 8,
        con_dim_head = 256,
        con_heads = 16,
    ).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        b = 16
        x = torch.randn(b, 64, 4, 64, 64).to(device)
        timesteps = torch.randint(0, 1000, (b,)).to(device)
        random_frame_indices = torch.randint(0, 64, (b,)).to(device)
        context_token = conditioner(x, timesteps, random_frame_indices)
        print(sum(p.numel() for p in conditioner.parameters()), torch.cuda.max_memory_allocated(device) / 1024 / 1024, "MB")
