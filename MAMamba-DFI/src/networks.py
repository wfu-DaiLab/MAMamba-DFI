import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange


try:
    from .vmamba import EfficientSS2D
except ImportError:
    print("Warning: VMamba modules not found. Please ensure vmamba.py is in the source directory.")


    # Placeholder for syntax checking if file is missing during dry run
    class SS2D(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()

        def forward(self, x): return x



class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask_in) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output




class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(EDFFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        # Frequency domain gating parameter
        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        b, c, h, w = x.shape
        # Pad if needed for patch FFT
        h_n = (self.patch_size - h % self.patch_size) % self.patch_size
        w_n = (self.patch_size - w % self.patch_size) % self.patch_size

        x_padded = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')


        x_patch = rearrange(x_padded, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
                            patch1=self.patch_size, patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))

        x_out = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
                          patch1=self.patch_size, patch2=self.patch_size)


        x_out = x_out[:, :, :h, :w]

        return x_out




class MaskAwareSSM(nn.Module):
    def __init__(self, dim, layer_idx=0):
        super(MaskAwareSSM, self).__init__()

        self.layer_idx = layer_idx


        self.ssm = EfficientSS2D(d_model=dim, d_state=32, ssm_ratio=2.0, d_conv=3,
                        dt_max=0.01, dt_min=0.0001, forward_type="v2")
        self.norm = LayerNorm(dim)

    def forward(self, x, mask):

        valid_gate = 1 - mask
        x_masked = x * valid_gate


        x_in = self.norm(x_masked)



        if self.layer_idx % 2 == 1:
            x_in = torch.flip(x_in, dims=(-2, -1)).contiguous()
        elif self.layer_idx % 2 == 0 and self.layer_idx > 0:  # Simple alternation logic
            x_in = torch.transpose(x_in, dim0=-2, dim1=-1).contiguous()


        out = self.ssm(x_in)


        if self.layer_idx % 2 == 1:
            out = torch.flip(out, dims=(-2, -1)).contiguous()
        elif self.layer_idx % 2 == 0 and self.layer_idx > 0:
            out = torch.transpose(out, dim0=-2, dim1=-1).contiguous()

        return out




class MSBlockLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expansion=2.0):
        super().__init__()
        hidden_dim = max(1, int(in_channels * expansion))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=1,
                      padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LocalMSBlock(nn.Module):
    def __init__(self, dim):
        super(LocalMSBlock, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim + 1, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )


        c1 = dim // 3
        c2 = dim // 3
        c3 = dim - c1 - c2
        self.split_sections = [c1, c2, c3]


        self.branch1 = MSBlockLayer(c1, c1, kernel_size=3)
        self.branch2 = MSBlockLayer(c2, c2, kernel_size=5)
        self.branch3 = MSBlockLayer(c3, c3, kernel_size=7)


        self.align_1_to_2 = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        self.align_2_to_3 = nn.Conv2d(c2, c3, 1, bias=False) if c2 != c3 else nn.Identity()


        self.out_conv = nn.Conv2d(dim, dim * 2, 1)
        self.norm = nn.InstanceNorm2d(dim * 2, affine=True)
        self.act = nn.ELU()

    def forward(self, x, mask):
        x_in = torch.cat([x, mask], dim=1)
        x_in = self.in_conv(x_in)

        x_splits = torch.split(x_in, self.split_sections, dim=1)
        x1, x2, x3 = x_splits[0], x_splits[1], x_splits[2]

        y1 = self.branch1(x1)


        y1_aligned = self.align_1_to_2(y1)
        y2 = self.branch2(x2 + y1_aligned)

        y2_aligned = self.align_2_to_3(y2)
        y3 = self.branch3(x3 + y2_aligned)


        y = torch.cat([y1, y2, y3], dim=1)
        out = self.out_conv(y)
        out = self.norm(out)
        feat, gate = out.chunk(2, dim=1)

        return x + (self.act(feat) * torch.sigmoid(gate))




class BiInteractionUnit(nn.Module):
    def __init__(self, dim):
        super(BiInteractionUnit, self).__init__()
        self.proj_global = nn.Conv2d(dim, dim, 1)
        self.proj_local = nn.Conv2d(dim, dim, 1)
        self.fusion = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, global_feat, local_feat):
        global_gate = torch.sigmoid(self.proj_global(global_feat))
        local_gated = local_feat * global_gate

        local_gate = torch.sigmoid(self.proj_local(local_feat))
        global_gated = global_feat * local_gate

        cat_feat = torch.cat([global_gated, local_gated], dim=1)
        out = self.fusion(cat_feat)
        return out




class DualStreamBlock(nn.Module):
    def __init__(self, dim, layer_idx=0):
        super(DualStreamBlock, self).__init__()


        self.global_branch = MaskAwareSSM(dim, layer_idx=layer_idx)


        self.local_branch = LocalMSBlock(dim)


        self.interaction = BiInteractionUnit(dim)


        self.edffn = EDFFN(dim)
        self.norm_ffn = LayerNorm(dim)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask):

        feat_global = self.global_branch(x, mask)
        feat_local = self.local_branch(x, mask)


        feat_fused = self.interaction(feat_global, feat_local)


        x = x + self.gamma * feat_fused


        x = x + self.edffn(self.norm_ffn(x))

        return x




class GatedEmb(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(GatedEmb, self).__init__()
        self.gproj1 = nn.Conv2d(in_c, embed_dim * 2, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.gproj1(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return x


class MaskAwareSDPDownsample(nn.Module):
    def __init__(self, in_dim, scale_factor=2, kernel_size=3):
        super().__init__()
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.to_q = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=scale_factor, padding=1, bias=False),
            # nn.LayerNorm([in_dim, 1, 1]) if False else nn.Identity(),
        )
        self.to_k = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.to_v = nn.Conv2d(in_dim, in_dim, 1, bias=False)

        self.gate = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.Sigmoid()
        )
        self.proj_out = nn.Conv2d(in_dim, in_dim * 2, 1, bias=False)
        self.mask_down = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)
        self.scale = in_dim ** -0.5

    def forward(self, x, mask):
        B, C, H, W = x.shape
        q = self.to_q(x)
        H_out, W_out = q.shape[2], q.shape[3]

        k = self.to_k(x)
        v = self.to_v(x)

        k_unfold = F.unfold(k, kernel_size=self.kernel_size, stride=self.scale_factor, padding=self.padding)
        v_unfold = F.unfold(v, kernel_size=self.kernel_size, stride=self.scale_factor, padding=self.padding)
        mask_unfold = F.unfold(mask, kernel_size=self.kernel_size, stride=self.scale_factor, padding=self.padding)

        q = rearrange(q, 'b c h w -> b (h w) 1 c')
        k_unfold = rearrange(k_unfold, 'b (c k_sq) l -> b l k_sq c', k_sq=self.kernel_size ** 2)
        v_unfold = rearrange(v_unfold, 'b (c k_sq) l -> b l k_sq c', k_sq=self.kernel_size ** 2)
        mask_unfold = rearrange(mask_unfold, 'b (1 k_sq) l -> b l 1 k_sq', k_sq=self.kernel_size ** 2)

        attn = (q @ k_unfold.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(mask_unfold > 0.5, -1e9)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v_unfold
        out = rearrange(out, 'b (h w) 1 c -> b c h w', h=H_out, w=W_out)

        g = self.gate(out)
        out = out * g
        out = self.proj_out(out)
        out_mask = self.mask_down(mask)

        return out, out_mask


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, mask):
        return self.body(x)


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )
        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)
        return outputs, [conv1, conv2, conv3, conv4, conv5]




class MAMamba-DFI(nn.Module):
    def __init__(self,
                 inp_channels=4,
                 out_channels=3,
                 dim=80,
                 num_blocks=[1, 2, 1, 2],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(MAMamba-DFI, self).__init__()

        self.patch_embed = GatedEmb(inp_channels, dim)



        curr_idx = 0
        self.encoder_level1 = nn.ModuleList()
        for i in range(num_blocks[0]):
            self.encoder_level1.append(DualStreamBlock(dim=dim, layer_idx=curr_idx))
            curr_idx += 1

        self.down1_2 = MaskAwareSDPDownsample(dim)

        self.encoder_level2 = nn.ModuleList()
        for i in range(num_blocks[1]):
            self.encoder_level2.append(DualStreamBlock(dim=int(dim * 2 ** 1), layer_idx=curr_idx))
            curr_idx += 1

        self.down2_3 = MaskAwareSDPDownsample(int(dim * 2 ** 1))

        self.encoder_level3 = nn.ModuleList()
        for i in range(num_blocks[2]):
            self.encoder_level3.append(DualStreamBlock(dim=int(dim * 2 ** 2), layer_idx=curr_idx))
            curr_idx += 1

        self.down3_4 = MaskAwareSDPDownsample(int(dim * 2 ** 2))


        self.latent = nn.ModuleList()
        for i in range(num_blocks[3]):
            self.latent.append(DualStreamBlock(dim=int(dim * 2 ** 3), layer_idx=curr_idx))
            curr_idx += 1


        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)


        self.decoder_level3 = nn.ModuleList()
        for i in range(num_blocks[2]):
            # Decoder continues the indexing or can reset?
            # EVS works best if diversity continues.
            self.decoder_level3.append(DualStreamBlock(dim=int(dim * 2 ** 2), layer_idx=curr_idx))
            curr_idx += 1

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)


        self.decoder_level2 = nn.ModuleList()
        for i in range(num_blocks[1]):
            self.decoder_level2.append(DualStreamBlock(dim=int(dim * 2 ** 1), layer_idx=curr_idx))
            curr_idx += 1

        self.up2_1 = Upsample(int(dim * 2 ** 1))


        self.decoder_level1 = nn.ModuleList()
        for i in range(num_blocks[0]):
            self.decoder_level1.append(DualStreamBlock(dim=int(dim * 2 ** 1), layer_idx=curr_idx))
            curr_idx += 1

        self.output = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    def forward_stage(self, stage_module_list, x, mask):
        for block in stage_module_list:
            x = block(x, mask)
        return x

    def forward(self, inp_img, mask_whole, mask_half, mask_quarter, mask_tiny):
        inp_enc_level1 = self.patch_embed(torch.cat((inp_img, mask_whole), dim=1))
        out_enc_level1 = self.forward_stage(self.encoder_level1, inp_enc_level1, mask_whole)

        inp_enc_level2, _ = self.down1_2(out_enc_level1, mask_whole)
        out_enc_level2 = self.forward_stage(self.encoder_level2, inp_enc_level2, mask_half)

        inp_enc_level3, _ = self.down2_3(out_enc_level2, mask_half)
        out_enc_level3 = self.forward_stage(self.encoder_level3, inp_enc_level3, mask_quarter)

        inp_enc_level4, _ = self.down3_4(out_enc_level3, mask_quarter)
        latent = self.forward_stage(self.latent, inp_enc_level4, mask_tiny)

        inp_dec_level3 = self.up4_3(latent, mask_tiny)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.forward_stage(self.decoder_level3, inp_dec_level3, mask_quarter)

        inp_dec_level2 = self.up3_2(out_dec_level3, mask_quarter)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.forward_stage(self.decoder_level2, inp_dec_level2, mask_half)

        inp_dec_level1 = self.up2_1(out_dec_level2, mask_half)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.forward_stage(self.decoder_level1, inp_dec_level1, mask_whole)

        out_dec_level1 = self.output(out_dec_level1)
        out_dec_level1 = (torch.tanh(out_dec_level1) + 1) / 2
        return out_dec_level1
