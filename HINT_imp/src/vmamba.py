import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from einops import rearrange, repeat

# 尝试导入官方 mamba_ssm
try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    print("Error: mamba_ssm not installed properly!")
    Mamba = None
    selective_scan_fn = None


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# ==========================================================
# EfficientSS2D: 单向扫描核心 (提速神器)
# Implements Single-Direction Selective Scan for EVS Strategy
# ==========================================================
class EfficientSS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            ssm_ratio=2.0,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # 2D 深度卷积 (Depthwise Conv)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # ▼▼▼ 关键修改：只初始化 1 组参数 (原本是 4 组) ▼▼▼
        # SSM 参数投影 (x -> dt, B, C)
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        # dt 投影 (x -> dt)
        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A 和 D 也只需要 1 份拷贝
        self.A_log = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.D = self.D_init(self.d_inner, copies=1, merge=True)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1)
        A_log = torch.log(A)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor, **kwargs):
        # Input: [B, C, H, W] -> Permute to [B, H, W, C] for Linear
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        # Linear Projection + Chunk -> x (Content), z (Gate)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Depthwise Conv
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, D_inner, H, W]
        x = self.act(self.conv2d(x))

        # ▼▼▼ 关键修改：只做单向扫描，不做 4 向展开 ▼▼▼
        # 1. Flatten to [B, L, D] (Sequence Length L = H*W)
        # Note: The input x here has already been geometrically transformed (flipped/transposed)
        # in `networks.py`, so we just scan it as a 1D sequence.
        y = x.view(B, self.d_inner, H * W).transpose(1, 2)  # [B, L, D]

        # 2. Generate Parameters (Delta, B, C)
        x_dbl = self.x_proj(y)  # [B, L, rank+2*state]
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dts = self.dt_proj(dts)  # [B, L, D]

        # 3. Reshape for selective_scan_fn
        # fn requires: [B, D, L], [B, D, L], [B, N, L] ...
        xs = y.transpose(1, 2).float().contiguous()  # [B, D, L]
        dts = dts.transpose(1, 2).float().contiguous()
        Bs = Bs.transpose(1, 2).float().contiguous()
        Cs = Cs.transpose(1, 2).float().contiguous()

        Ds = self.D.float().view(-1)
        As = -torch.exp(self.A_log.float())
        dt_projs_bias = self.dt_proj.bias.float()

        # 4. Execute Single-Direction Selective Scan
        out_y = selective_scan_fn(
            xs, dts,
            As, Bs, Cs, Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False
        )  # Output: [B, D, L]

        # 5. Restore Shape: [B, D, L] -> [B, H, W, D]
        # (Transposed back to channels-last for LayerNorm)
        y = out_y.transpose(1, 2).view(B, H, W, self.d_inner)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # Norm & Gating
        y = self.out_norm(y)
        y = y * F.silu(z)  # Gating with z branch

        # Output Projection
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        # Restore to [B, C, H, W]
        return out.permute(0, 3, 1, 2)


# ==========================================================
# Legacy SS2D: 4-Direction Scan (Retained for compatibility)
# ==========================================================
# This class is kept just in case you want to switch back to full VMamba mode.
# But networks.py will now import EfficientSS2D by default.
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            ssm_ratio=2.0,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner,
                                bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs)
        self.act = nn.SiLU()

        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(4)
        ])

        self.dt_projs = nn.ModuleList([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(4)
        ])

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1)
        A_log = torch.log(A)
        if copies > 1: A_log = repeat(A_log, "d n -> r d n", r=copies)
        if merge: A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log);
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1: D = repeat(D, "d1 -> r d1", r=copies)
        if merge: D = D.flatten(0, 1)
        D = nn.Parameter(D);
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor, **kwargs):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y1 = x.view(B, -1, H * W)
        y2 = torch.flip(y1, dims=[-1])
        y3 = torch.transpose(x, 2, 3).contiguous().view(B, -1, H * W)
        y4 = torch.flip(y3, dims=[-1])
        y = torch.stack([y1, y2, y3, y4], dim=1).transpose(2, 3)

        x_proj_w = torch.stack([t.weight for t in self.x_proj], dim=0)
        x_dbl = torch.einsum("b k l d, k c d -> b k l c", y, x_proj_w)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_projs_w = torch.stack([t.weight for t in self.dt_projs], dim=0)
        dt_projs_b = torch.stack([t.bias for t in self.dt_projs], dim=0)
        dts = torch.einsum("b k l r, k d r -> b k l d", dts, dt_projs_w)

        xs = y.float().contiguous().view(B, -1, 4 * self.d_inner)
        dts = dts.contiguous().float().view(B, -1, 4 * self.d_inner)
        Bs = Bs.float().contiguous().view(B, -1, 4 * self.d_state)
        Cs = Cs.float().contiguous().view(B, -1, 4 * self.d_state)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float())
        dt_projs_bias = dt_projs_b.float().view(-1)

        out_y = selective_scan_fn(
            xs.transpose(1, 2), dts.transpose(1, 2), As,
            Bs.transpose(1, 2).view(B, 4, -1, H * W), Cs.transpose(1, 2).view(B, 4, -1, H * W), Ds,
            z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False
        )
        out_y = out_y.transpose(1, 2).view(B, H * W, 4, -1)

        y1_o = out_y[:, :, 0, :]
        y2_o = torch.flip(out_y[:, :, 1, :], dims=[1])
        y3_o = out_y[:, :, 2, :].view(B, W, H, -1).transpose(1, 2).reshape(B, H * W, -1)
        y4_o = torch.flip(out_y[:, :, 3, :].view(B, W, H, -1).transpose(1, 2).reshape(B, H * W, -1), dims=[1])

        y = y1_o + y2_o + y3_o + y4_o
        y = self.out_norm(y).view(B, H, W, -1)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None: out = self.dropout(out)
        return out.permute(0, 3, 1, 2)


class VSSBlock(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0, norm_layer: nn.Module = nn.LayerNorm,
                 channel_first=False, ssm_d_state: int = 16, ssm_ratio=2.0, ssm_dt_rank: any = "auto",
                 ssm_act_layer=nn.SiLU,
                 ssm_conv: int = 3, ssm_conv_bias=True, ssm_drop_rate: float = 0, ssm_init="v0", forward_type="v2",
                 mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate: float = 0.0, gmlp=False,
                 use_checkpoint: bool = False, **kwargs):
        super().__init__()
        # VSSBlock 保留作为接口，默认使用 EfficientSS2D (如果通过参数控制)
        # 但在你的 DualStreamBlock 中，你是直接调用的 SS2D 或 EfficientSS2D，所以这个类其实可能用不到
        # 这里为了完整性保留
        self.norm = norm_layer(hidden_dim)
        self.op = EfficientSS2D(  # 默认改为 Efficient
            d_model=hidden_dim, d_state=ssm_d_state, ssm_ratio=ssm_ratio, dt_rank=ssm_dt_rank,
            d_conv=ssm_conv, conv_bias=ssm_conv_bias, dropout=ssm_drop_rate, bias=False,
        )

    def forward(self, input: torch.Tensor):
        return input + self.op(self.norm(input))