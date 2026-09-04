"""
Shared spatial building blocks vendored from the SuperDARN Pangu architecture.

These are the geometry-aware transformer primitives (EarthSpecificBlock,
EarthAttention3D, PatchEmbed2D, PatchRecovery2D, DownSample, UpSample) used
by the SuperDARNEncoder.  They contain NO solar-wind conditioning — FiLM layers
have been removed entirely and live in dynamics.py instead.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Patch embed / recovery
# ---------------------------------------------------------------------------

class _Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class PatchEmbed2D(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        H, W = img_size
        ph, pw = patch_size
        pt = (ph - H % ph) % ph
        pl = (pw - W % pw) % pw
        layers = [
            nn.ZeroPad2d((pl // 2, pl - pl // 2, pt // 2, pt - pt // 2)),
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
        ]
        if norm_layer is not None:
            layers += [_Permute(0, 2, 3, 1), norm_layer(embed_dim), _Permute(0, 3, 1, 2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchRecovery2D(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)
        H, W = img_size
        ph, pw = patch_size
        out_H = -(-H // ph) * ph
        out_W = -(-W // pw) * pw
        pt = (out_H - H) // 2
        pb = out_H - H - pt
        pl = (out_W - W) // 2
        pr = out_W - W - pl
        self._crop = (
            slice(None), slice(None),
            slice(pt, out_H - pb if pb else None),
            slice(pl, out_W - pr if pr else None),
        )

    def forward(self, x):
        return self.conv(x)[self._crop]


# ---------------------------------------------------------------------------
# Down / up sample
# ---------------------------------------------------------------------------

class DownSample(nn.Module):
    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        in_pl, in_lat, in_lon = input_resolution
        out_pl, out_lat, out_lon = output_resolution
        assert in_pl == out_pl
        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon
        pt, pb = h_pad // 2, h_pad - h_pad // 2
        pl, pr = w_pad // 2, w_pad - w_pad // 2
        self._fpad = (0, 0, pl, pr, pt, pb, 0, 0)
        self._in = (in_pl, in_lat, in_lon)
        self._out = (out_pl, out_lat, out_lon)
        self._out_n = out_pl * out_lat * out_lon

    def forward(self, x):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self._in
        out_pl, out_lat, out_lon = self._out
        x = x.reshape(B, in_pl, in_lat, in_lon, C)
        x = F.pad(x, self._fpad)
        x = (x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C)
              .permute(0, 1, 2, 4, 3, 5, 6)
              .reshape(B, self._out_n, 4 * C))
        return self.linear(self.norm(x))


class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        in_pl, in_lat, in_lon = input_resolution
        out_pl, out_lat, out_lon = output_resolution
        assert in_pl == out_pl
        pt = (in_lat * 2 - out_lat) // 2
        pb = (in_lat * 2 - out_lat) - pt
        pl = (in_lon * 2 - out_lon) // 2
        pr = (in_lon * 2 - out_lon) - pl
        self._in = (in_pl, in_lat, in_lon)
        self._crop = (
            slice(None), slice(None, out_pl),
            slice(pt, 2 * in_lat - pb if pb else None),
            slice(pl, 2 * in_lon - pr if pr else None),
            slice(None),
        )

    def forward(self, x):
        B, _, C = x.shape
        in_pl, in_lat, in_lon = self._in
        x = self.linear1(x)
        x = (x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2)
              .permute(0, 1, 2, 4, 3, 5, 6)
              .reshape(B, in_pl, in_lat * 2, in_lon * 2, -1))
        return self.linear2(self.norm(x[self._crop].reshape(B, -1, x.shape[-1])))


# ---------------------------------------------------------------------------
# Window attention helpers
# ---------------------------------------------------------------------------

def _get_pad3d(input_resolution, window_size):
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size
    pl = (win_pl  - Pl  % win_pl)  % win_pl
    la = (win_lat - Lat % win_lat) % win_lat
    lo = (win_lon - Lon % win_lon) % win_lon
    return (lo // 2, lo - lo // 2, la // 2, la - la // 2, pl // 2, pl - pl // 2)


def _get_earth_position_index(window_size):
    win_pl, win_lat, win_lon = window_size
    zi = torch.arange(win_pl)
    zj = -torch.arange(win_pl) * win_pl
    hi = torch.arange(win_lat)
    hj = -torch.arange(win_lat) * win_lat
    w  = torch.arange(win_lon)
    c1 = torch.stack(torch.meshgrid(zi, hi, w, indexing='ij'))
    c2 = torch.stack(torch.meshgrid(zj, hj, w, indexing='ij'))
    coords = (c1.flatten(1)[:, :, None] - c2.flatten(1)[:, None, :]).permute(1, 2, 0)
    coords[:, :, 2] += win_lon - 1
    coords[:, :, 1] *= 2 * win_lon - 1
    coords[:, :, 0] *= (2 * win_lon - 1) * win_lat * win_lat
    return coords.sum(-1)


def _get_shift_window_mask(input_resolution, window_size, shift_size):
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size
    sh_pl, sh_lat, sh_lon = shift_size
    mask = torch.zeros(1, Pl, Lat, Lon + sh_lon, 1)
    pl_s  = (slice(0, -win_pl),  slice(-win_pl,  -sh_pl  or None), slice(-sh_pl,  None))
    lat_s = (slice(0, -win_lat), slice(-win_lat, -sh_lat or None), slice(-sh_lat, None))
    lon_s = (slice(0, -win_lon), slice(-win_lon, -sh_lon or None), slice(-sh_lon, None))
    cnt = 0
    for p in pl_s:
        for la in lat_s:
            for lo in lon_s:
                mask[:, p, la, lo, :] = cnt
                cnt += 1
    mask = mask[:, :, :, :Lon, :]
    # partition into windows → (n_lon, n_pl*n_lat, win_vol)
    B, P, La, Lo, C = mask.shape
    wP, wLa, wLo = window_size
    w = mask.view(B, P // wP, wP, La // wLa, wLa, Lo // wLo, wLo, C)
    w = w.permute(0, 5, 1, 3, 2, 4, 6, 7).reshape(-1, (P // wP) * (La // wLa), wP * wLa * wLo)
    attn = w.unsqueeze(2) - w.unsqueeze(3)
    return attn.masked_fill(attn != 0, -100.0).masked_fill(attn == 0, 0.0)


class _WindowPartition(nn.Module):
    def __init__(self, input_shape, window_size):
        super().__init__()
        B, Pl, Lat, Lon, C = input_shape
        wP, wLa, wLo = window_size
        self._view  = (-1, (Pl // wP) * (Lat // wLa), wP * wLa * wLo, C)
        self._xview = (-1, Pl // wP, wP, Lat // wLa, wLa, Lon // wLo, wLo, C)

    def forward(self, x):
        return x.view(*self._xview).permute(0, 5, 1, 3, 2, 4, 6, 7).reshape(*self._view)


class _WindowReverse(nn.Module):
    def __init__(self, window_size, Pl, Lat, Lon, C):
        super().__init__()
        wP, wLa, wLo = window_size
        self._wview = (-1, Lon // wLo, Pl // wP, Lat // wLa, wP, wLa, wLo, C)
        self._xview = (-1, Pl, Lat, Lon, C)

    def forward(self, w):
        return w.unflatten(2, (w.shape[2],)).view(*self._wview).permute(0, 2, 4, 3, 5, 1, 6, 7).reshape(*self._xview)


class _Crop3D(nn.Module):
    def __init__(self, padding):
        super().__init__()
        pf, pb = padding[-1], padding[-2]
        pt, pbo = padding[2], padding[3]
        pl, pr = padding[0], padding[1]
        self._crop = (
            slice(None),
            slice(pf, -pb  if pb  else None),
            slice(pt, -pbo if pbo else None),
            slice(pl, -pr  if pr  else None),
            slice(None),
        )

    def forward(self, x):
        return x[self._crop]


# ---------------------------------------------------------------------------
# DropPath
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self._scale = 1.0 / (1.0 - drop_prob) if drop_prob < 1.0 else 1.0

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(1 - self.drop_prob)
        return x * keep * self._scale


# ---------------------------------------------------------------------------
# EarthAttention3D  (no FiLM — pure spatial attention)
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


class EarthAttention3D(nn.Module):
    def __init__(self, dim, input_resolution, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0,
                 attn_mask=None):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        n_types = (input_resolution[0] // window_size[0]) * (input_resolution[1] // window_size[1])
        win_vol = window_size[0] * window_size[1] * window_size[2]
        bias_table = torch.zeros(win_vol ** 2 // window_size[2] * (2 * window_size[2] - 1),
                                 n_types, num_heads)
        # Truncated-normal init
        std = 0.02
        l, u = _norm_cdf(-2 / std), _norm_cdf(2 / std)
        bias_table.uniform_(2 * l - 1, 2 * u - 1).erfinv_().mul_(std * math.sqrt(2)).clamp_(-2, 2)

        pos_idx = _get_earth_position_index(window_size)
        self.register_buffer("pos_idx", pos_idx)

        bias = bias_table[pos_idx.view(-1)].view(win_vol, win_vol, n_types, num_heads)
        bias = bias.permute(3, 2, 0, 1).contiguous().unsqueeze(0)
        self.earth_pos_bias = nn.Parameter(bias)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        if attn_mask is not None:
            self.register_buffer("attn_mask", attn_mask)
            self._nLon = attn_mask.shape[0]
            self._masked = True
        else:
            self._masked = False

    def forward(self, x):
        B_, nW_, N, C = x.shape
        hd = C // self.num_heads
        Bn = B_ * nW_

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, hd)).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv.unbind(0)

        bias = self.earth_pos_bias.expand(B_, -1, -1, -1, -1)
        if self._masked:
            bias = (bias.view(-1, self._nLon, self.num_heads, nW_, N, N)
                        + self.attn_mask.unsqueeze(1).unsqueeze(0))
            bias = bias.view(-1, self.num_heads, nW_, N, N)
        bias = bias.permute(0, 2, 1, 3, 4).reshape(Bn, self.num_heads, N, N)

        q = q.permute(0, 2, 1, 3, 4).reshape(Bn, self.num_heads, N, hd)
        k = k.permute(0, 2, 1, 3, 4).reshape(Bn, self.num_heads, N, hd)
        v = v.permute(0, 2, 1, 3, 4).reshape(Bn, self.num_heads, N, hd)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )
        out = out.permute(0, 2, 1, 3).reshape(B_, nW_, N, C)
        return self.proj_drop(self.proj(out))


# ---------------------------------------------------------------------------
# EarthSpecificBlock  (no FiLM — pure transformer block)
# ---------------------------------------------------------------------------

class EarthSpecificBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads,
                 window_size=(2, 8, 16), shift_size=(0, 0, 0),
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution

        padding = _get_pad3d(input_resolution, window_size)
        self._fpad = (0, 0) + padding
        pad_res = [
            input_resolution[0] + padding[-1] + padding[-2],
            input_resolution[1] + padding[2]  + padding[3],
            input_resolution[2] + padding[0]  + padding[1],
        ]

        self.norm1 = norm_layer(dim)
        self.wp  = _WindowPartition((2, *pad_res, dim), window_size)
        self.wr  = _WindowReverse(window_size, *pad_res, dim)
        self.c3d = _Crop3D(padding)

        roll = any(s != 0 for s in shift_size)
        self._neg = (-shift_size[0], -shift_size[1], -shift_size[2]) if roll else None
        self._pos = shift_size if roll else None
        attn_mask = _get_shift_window_mask(pad_res, window_size, shift_size) if roll else None

        self.attn = EarthAttention3D(
            dim=dim, input_resolution=pad_res, window_size=window_size,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_mask=attn_mask,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x).unflatten(1, self.input_resolution)
        x = F.pad(x, self._fpad)
        if self._neg:
            x = torch.roll(x, shifts=self._neg, dims=(1, 2, 3))
        x = self.wr(self.attn(self.wp(x)))
        if self._pos:
            x = torch.roll(x, shifts=self._pos, dims=(1, 2, 3))
        x = self.c3d(x).flatten(1, 3)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))
