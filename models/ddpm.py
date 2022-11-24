"""
DDPM Model

This code is the pytorch equivalent of
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
import torch
import torch.nn as nn
import functools

from . import utils, layers

ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
get_act = layers.get_act
get_normalization = layers.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='ddpm')
class DDPM(nn.Module):
    """ The DDPM network implements a U-Net based on a Wide ResNet
        with weight normalization using group normalization.

        Self-attention blocks at the 16x16 resolution
        Transformer sinusoidal position embedding specifies the time dependency $t$

    Variable's descriptions:
        nf: embedding size for Fourier transform
    """
    def __init__(self, config):
        super().__init__()
        self.act = act = get_act(config)
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

        AttnBlock = functools.partial(layers.AttnBlock)
        self.conditional = conditional = config.model.conditional
        ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
        if conditional:
            # Condition on noise levels
            modules = [nn.Linear(nf, nf * 4)]
            modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)

        self.centered = config.data.centered
        channels = config.data.num_channels

        # Downsampling block
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
                hs_c.append(in_ch)
        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))
        # print(hs_c)
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
            if i_level != 0:
                modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
        modules.append(conv3x3(in_ch, channels, init_scale=0.))
        self.all_modules = nn.ModuleList(modules)

        self.scale_by_sigma = config.model.scale_by_sigma

    def forward(self, x, labels):
        """
        :param x: perturbed sample
        :param labels: the time steps per batch
        :return:
        """
        modules = self.all_modules
        m_idx = 0
        if self.conditional:
            # timestep/scale embedding
            timesteps = labels
            temb = layers.get_timestep_embedding(timesteps, self.nf)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if self.centered:
            # Input is in [-1, 1]
            h = x
        else:
            # Input is in [0,1]
            h = 2 * x - 1.
        # Downsampling block
        hs = [modules[m_idx](h)]
        # print(m_idx, hs[-1].shape)
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                # print(m_idx, i_level, i_block, hs[-1].shape)
                m_idx += 1
                # run attention if the resolution is as specified in `Config`
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    # print(m_idx, i_level, i_block, hs[-1].shape)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                # MaxPool 2x2
                hs.append(modules[m_idx](hs[-1]))
                # print("Max pool", m_idx, i_level, i_block, hs[-1].shape)
                m_idx += 1

        h = hs[-1]
        h = modules[m_idx](h, temb)
        # print(m_idx, h[-1].shape)
        m_idx += 1
        h = modules[m_idx](h)
        # print(m_idx, h[-1].shape)
        m_idx += 1
        h = modules[m_idx](h, temb)
        # print(m_idx, h[-1].shape)
        m_idx += 1
        # print("shape", len(hs), hs[0].shape)
        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                # print(m_idx, i_level, i_block, h.shape)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    # print(m_idx, i_level, i_block, h.shape)
                    m_idx += 1
            if i_level != 0:
                # Up Convolution
                h = modules[m_idx](h)
                # print(m_idx, i_level, i_block, h.shape)
                m_idx += 1

        assert not hs
        h = self.act(modules[m_idx](h))
        # print(m_idx, i_level, i_block, h.shape)
        m_idx += 1
        h = modules[m_idx](h)
        # print(m_idx, i_level, i_block, h.shape)
        m_idx += 1
        assert m_idx == len(modules)

        if self.scale_by_sigma:
            # Divide the output by sigmas. Useful for training with the NCSN loss.
            # The DDPM loss scales the network output by sigma in the loss function,
            # so no need of doing it here.
            used_sigmas = self.sigmas[labels, None, None, None]
            h = h / used_sigmas

        return h


### Debuging the network
@utils.register_model(name='ddpm_debugging')
class DDPM_features(nn.Module):
    """ The DDPM network implements a U-Net based on a Wide ResNet
        with weight normalization using group normalization.

        Self-attention blocks at the 16x16 resolution
        Transformer sinusoidal position embedding specifies the time dependency $t$

    Variable's descriptions:
        nf: embedding size for Fourier transform
    """
    def __init__(self, config, flag_print=False):
        super().__init__()
        self.print = flag_print
        self.act = act = get_act(config)
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

        AttnBlock = functools.partial(layers.AttnBlock)
        self.conditional = conditional = config.model.conditional
        ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
        if conditional:
            # Condition on noise levels
            modules = [nn.Linear(nf, nf * 4)]
            modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)

        self.centered = config.data.centered
        channels = config.data.num_channels

        # Downsampling block
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
                hs_c.append(in_ch)
        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))
        if self.print: print("wtf", hs_c)
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
            if i_level != 0:
                modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
        modules.append(conv3x3(in_ch, channels, init_scale=0.))
        self.all_modules = nn.ModuleList(modules)

        self.scale_by_sigma = config.model.scale_by_sigma

    def forward(self, x, labels):
        """
        :param x: perturbed sample
        :param labels: the time steps per batch
        :return:
        """
        # save the features to build the perceptual loss
        features=[]
        modules = self.all_modules
        m_idx = 0
        if self.conditional:
            # timestep/scale embedding
            timesteps = labels
            temb = layers.get_timestep_embedding(timesteps, self.nf)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if self.centered:
            # Input is in [-1, 1]
            h = x
        else:
            # Input is in [0,1]
            h = 2 * x - 1.

        if self.print: print("Input image", h.shape)
        # Downsampling block
        hs = [modules[m_idx](h)]
        if self.print: print(m_idx, hs[-1].shape)
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                if self.print: print(m_idx, i_level, i_block, h.shape)
                m_idx += 1
                # run attention if the resolution is as specified in `Config`
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    if self.print: print(m_idx, i_level, i_block, h.shape)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                # save one before downsampling
                features.append(hs[-1])
                # MaxPool 2x2
                hs.append(modules[m_idx](hs[-1]))
                if self.print: print("Max pool", m_idx, i_level, i_block, hs[-1].shape)
                m_idx += 1
        h = hs[-1]
        h = modules[m_idx](h, temb)
        if self.print: print("Last level", m_idx, h.shape)
        m_idx += 1
        h = modules[m_idx](h)
        features.append(h) # save after attention block
        if self.print: print(m_idx, h.shape)
        m_idx += 1
        h = modules[m_idx](h, temb)
        if self.print: print(m_idx, h.shape)
        m_idx += 1
        if self.print: print("shape", len(hs), hs[0].shape)
        if self.print: print("Upsampling")
        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                if self.print: print(m_idx, i_level, i_block, h.shape)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    if self.print: print(m_idx, i_level, i_block, h.shape)
                    m_idx += 1
            if i_level != 0:
                # Up Convolution
                h = modules[m_idx](h)
                features.append(h)
                if self.print: print(m_idx, i_level, i_block, h.shape)
                m_idx += 1

        assert not hs
        h = self.act(modules[m_idx](h))
        if self.print: print("After activation ",m_idx, i_level, i_block, h.shape)
        features.append(h)
        m_idx += 1
        h = modules[m_idx](h)
        if self.print: print("one more", m_idx, i_level, i_block, h.shape)
        m_idx += 1
        assert m_idx == len(modules)
        
        if self.scale_by_sigma:
            # Divide the output by sigmas. Useful for training with the NCSN loss.
            # The DDPM loss scales the network output by sigma in the loss function,
            # so no need of doing it here.
            used_sigmas = self.sigmas[labels, None, None, None]
            h = h / used_sigmas

        return h, features




