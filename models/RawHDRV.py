import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
from scipy.signal.windows import gaussian
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer, LayerNorm
from .spynet_arch import SpyNet
from einops import rearrange
import numbers
from data.process import process
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, withmask=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.withmask = withmask
        if withmask:
            self.max_dsc = nn.AdaptiveMaxPool1d(1)
            self.avg_dsc = nn.AdaptiveAvgPool1d(1)
            self.linear1 = nn.Linear(2, 1, bias=bias)
            self.linear2 = nn.Linear(dim//num_heads, dim//num_heads, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
    
            
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature # b, h, c, c
        
        if self.withmask:
            mask_in = attn.clone()
            b_, h_, c_, _ = mask_in.shape 
            dsc = torch.cat([self.max_dsc(mask_in.reshape(b_, h_*c_, c_)), self.avg_dsc(mask_in.reshape((b_, h_*c_, c_)))], dim=-1).reshape(b_, h_, c_, 2) # b,h,c,2
            mask = self.linear1(dsc) # b,h,c,1
            mask = self.linear2(F.gelu(mask).transpose(-2,-1)).transpose(-2,-1) # b, h, c, 1
            mask = F.sigmoid(mask)
        
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        if self.withmask:
            out = out * mask
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, 
                 masked=False, squeeze_factor=1.0, ch_compress=False):
        super(TransformerBlock, self).__init__()
        if ch_compress:
            self.norm1 = LayerNorm(dim//squeeze_factor, LayerNorm_type)
            self.attn = Attention(dim//squeeze_factor, num_heads, bias, masked)
        else:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = Attention(dim, num_heads, bias, masked)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.ch_compress = ch_compress and (squeeze_factor > 1)
        if self.ch_compress:
            self.compress = nn.Conv2d(dim, dim//squeeze_factor, kernel_size=3, stride=1, padding=1, bias=bias)
            self.expand = nn.Conv2d(dim//squeeze_factor, dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        if self.ch_compress:
            x_compressed = self.compress(x)
            x = x +  self.expand(self.attn(self.norm1(x_compressed)))
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, alpha=0.5):
        """
        Dual Branch Attention with Self and Cross Attention
        
        Args:
            dim: input dimension
            num_heads: number of attention heads
            bias: whether to use bias
            withmask: whether to use channel attention mask
            attention_mode: 'self', 'cross', or 'dual'
            alpha: mixing ratio for dual branch (0.0-1.0)
        """
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.alpha = alpha
        
        # Temperature parameters for both branches
        self.temperature_self = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_cross = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_cross = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv_cross = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.q_dwconv_cross = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_dwconv_cross = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out_cross = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def cross_attention(self, x1, x2):
        """Cross-attention branch: x1 as query, x2 as key-value"""
        b, c, h, w = x1.shape
        
        # Query from x1, Key-Value from x2
        q = self.q_dwconv_cross(self.q_cross(x1))
        kv = self.kv_dwconv_cross(self.kv_cross(x2))
        k, v = kv.chunk(2, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature_cross
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out_cross(out)
        
        return out

    def forward(self, x, x_ref=None):
        """
        Forward pass
        
        Args:
            x: main input feature
            x_ref: reference feature for cross-attention (optional)
        """
        return self.cross_attention(x, x_ref)
        

##########################################################################
## Enhanced Transformer Block with Dual Branch Attention
class CrossAttentionTransformer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, alpha=0.5,
                 squeeze_factor=1.0, ch_compress=False):
        super(CrossAttentionTransformer, self).__init__()
        
        if ch_compress:
            self.norm1 = LayerNorm(dim//squeeze_factor, LayerNorm_type) 
            self.attn = CrossAttention(dim//squeeze_factor, num_heads, bias, alpha)
        else:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = CrossAttention(dim, num_heads, bias, alpha)
        
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        
        self.ch_compress = ch_compress and (squeeze_factor > 1)
        if self.ch_compress:
            self.compress = nn.Conv2d(dim, dim//squeeze_factor, kernel_size=3, stride=1, padding=1, bias=bias)
            self.expand = nn.Conv2d(dim//squeeze_factor, dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x, x_ref=None):
        if self.ch_compress:
            x_compressed = self.compress(x)
            x_ref_compressed = self.compress(x_ref) if x_ref is not None else None
            x = x + self.expand(self.attn(self.norm1(x_compressed), x_ref_compressed))
        else:
            x = x + self.attn(self.norm1(x), x_ref)
        
        x = x + self.ffn(self.norm2(x))
        return x


class GaussianBlur(nn.Module):

    def __init__(self, kernel_size=15, std=2.0):
        super(GaussianBlur, self).__init__()
        
        def get_kernel(size, std):
            """Returns a 2D Gaussian kernel array."""
            k = gaussian(size, std=std).reshape(size, 1)
            k = np.outer(k, k)
            return k / k.sum()

        kernel = get_kernel(kernel_size, std)
        kernel = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer('kernel', kernel)
        self.padding = kernel_size // 2
    
    def forward(self, mask):
        """Blur exposure mask. Supports [B,H,W] or [B,1,H,W]."""
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        softened_mask = F.conv2d(mask, self.kernel, padding=self.padding)

        return torch.clamp(softened_mask, 0, 1)


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class RawHDRV(nn.Module):
    """A recurrent network for video SR with complementary mask-based feature enhancement.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, RB_gudie=True, G_guidance=True, mask_guide=True,  softmask=True, softblending=False, 
                 num_blocks=[], spynet_path=None, heads=[], ffn_expansion_factor=2, masked=False,
                 bias=False, LayerNorm_type='BiasFree', ch_compress=False, squeeze_factor=[1,1,1]):
        super().__init__()
        # embedding
        self.num_feat = num_feat
        self.softmask = softmask
        self.softblending = softblending

        in_channels = 4
        out_channels = 4
        g_channels = 2
        
        self.RB_guidance = RB_gudie
        self.G_guidance = G_guidance
        self.mask_guide = mask_guide

        if self.RB_guidance:
            self.rb_feature_extractor = nn.Sequential(
                nn.Conv2d(2, num_feat, 3, 1, 1),
                TransformerBlock(dim=num_feat, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, 
                               bias=bias, LayerNorm_type=LayerNorm_type, masked=False, ch_compress=False, 
                               squeeze_factor=squeeze_factor[0])
            )
        
        if self.G_guidance:
            self.g_feature_extractor = nn.Sequential(
                nn.Conv2d(2, num_feat, 3, 1, 1),
                TransformerBlock(dim=num_feat, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, 
                               bias=bias, LayerNorm_type=LayerNorm_type, masked=False, ch_compress=False, 
                               squeeze_factor=squeeze_factor[0])
            )

        self.shallow_extraction = nn.Sequential(nn.Conv2d(4, num_feat, 3, 1, 1),
                                                TransformerBlock(dim=num_feat, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, masked=False, ch_compress=False, squeeze_factor=squeeze_factor[0]))
        
        fusion_input_channels = num_feat
        if self.RB_guidance:
            fusion_input_channels += num_feat
        if self.G_guidance:
            fusion_input_channels += num_feat
            
        self.multi_channel_fusion = nn.Sequential(
            nn.Conv2d(fusion_input_channels*2, num_feat, 3, 1, 1, bias=bias)
        )
        
        # alignment
        self.spynet = SpyNet(spynet_path)
        self.attn_merge = CrossAttentionTransformer(dim=num_feat, num_heads=1,ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type, ch_compress=ch_compress, squeeze_factor=4)
        
        # propagation
        # self.fusion = DualBranchTransformerBlock(dim=num_feat, num_heads=1,ffn_expansion_factor=ffn_expansion_factor,bias=bias, attention_mode='dual', LayerNorm_type=LayerNorm_type, masked=masked, ch_compress=ch_compress, squeeze_factor=4)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=num_feat, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, masked=masked,ch_compress=ch_compress, squeeze_factor=squeeze_factor[0]) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(num_feat) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(num_feat*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, masked=masked,ch_compress=ch_compress, squeeze_factor=squeeze_factor[0]) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(num_feat*2**1)) ## From Level 2 to Level 3
        self.latent= nn.Sequential(*[TransformerBlock(dim=int(num_feat*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, masked=masked,ch_compress=ch_compress, squeeze_factor=squeeze_factor[0]) for i in range(num_blocks[2])])
        self.up3_2 = Upsample(int(num_feat*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(num_feat*2**2), int(num_feat*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(num_feat*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, masked=masked,ch_compress=ch_compress, squeeze_factor=squeeze_factor[0]) for i in range(num_blocks[1])])
        self.up2_1 = Upsample(int(num_feat*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(num_feat*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, masked=masked,ch_compress=ch_compress, squeeze_factor=squeeze_factor[0]) for i in range(num_blocks[0])])
        self.reduce_chan_level1 = nn.Conv2d(int(num_feat*2**1), int(num_feat), kernel_size=1, bias=bias)
        self.refinement = nn.Sequential(*[TransformerBlock(dim=num_feat, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, masked=masked,ch_compress=ch_compress, squeeze_factor=squeeze_factor[0]) for i in range(num_blocks[3])])
        
        # reconstruction
        self.conv_hr = nn.Conv2d(num_feat, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 4, 3, 1, 1)
        self.blur = GaussianBlur().cuda()

        self.mask = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1), ResidualBlock_noBN(32), ResidualBlock_noBN(32), nn .Conv2d(32, 2, 3, 1, 1), nn.Sigmoid())

        self.l2 = nn.Sequential(nn.Conv2d(4, num_feat//2, kernel_size=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.PixelUnshuffle(2))
        
        self.l3 = nn.Sequential(nn.Conv2d(4, num_feat//4, kernel_size=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.PixelUnshuffle(4))
        
        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
       
        self.over_threshold = 0.5
        
    def compute_channel_specific_mask(self, channel_data):
        """Compute over/under-exposure masks for one channel group."""
        if self.softmask:
            if self.softblending:
                max_channel = channel_data.max(dim=1)[0]        # [B,H,W]
                min_channel = channel_data.min(dim=1)[0]        # [B,H,W]
                mask_over = (max_channel - 0.8).clamp(min=0.0, max=1.0) / 0.2
                mask_under = (0.2 - min_channel).clamp(min=0.0, max=1.0) / 0.2
            else:
                mask = self.mask(channel_data)
                mask_over = mask[:, 0] 
                mask_under = mask[:, 1] 
        else:
            max_channel = torch.max(channel_data, dim=1)[0]
            min_channel = torch.min(channel_data, dim=1)[0]
            mask_over = torch.zeros_like(max_channel)
            mask_over[max_channel > 0.95] = 1
            mask_under = torch.zeros_like(min_channel)
            mask_under[min_channel < 0.05] = 1
            mask_over = self.blur(mask_over).squeeze(1)
            mask_under = self.blur(mask_under).squeeze(1)
        
        return mask_over, mask_under
    
    def get_flow_bidirectional(self, x):
        """Compute bidirectional optical flow."""
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        
        return flows_forward, flows_backward

    def complementary_feature_enhancement(self, feat1, feat2, mask1_over, mask2_over):
        """Enhance pairwise features based on complementary exposure masks."""
        mask1_binary = (mask1_over > self.over_threshold).float()
        mask2_binary = (mask2_over > self.over_threshold).float()

        both_over = mask1_binary * mask2_binary
        frame1_normal_frame2_over = (1 - mask1_binary) * mask2_binary
        frame1_over_frame2_normal = mask1_binary * (1 - mask2_binary)

        weight1 = torch.ones_like(mask1_binary)
        weight2 = torch.ones_like(mask2_binary)

        normal_enhance_factor = 5.0
        over_suppress_factor = 0.1
        
        weight1 = weight1 + frame1_normal_frame2_over * (normal_enhance_factor - 1.0)
        weight2 = weight2 * (1 - frame1_normal_frame2_over * (1 - over_suppress_factor))

        weight1 = weight1 * (1 - frame1_over_frame2_normal * (1 - over_suppress_factor))
        weight2 = weight2 + frame1_over_frame2_normal * (normal_enhance_factor - 1.0)

        both_over_suppress = 0.1
        weight1 = weight1 * (1 - both_over * (1 - both_over_suppress))
        weight2 = weight2 * (1 - both_over * (1 - both_over_suppress))

        feat1_enhanced = feat1
        feat2_enhanced = feat2

        feat1_processed = feat1_enhanced * weight1.unsqueeze(1).expand_as(feat1_enhanced)
        feat2_processed = feat2_enhanced * weight2.unsqueeze(1).expand_as(feat2_enhanced)
        
        return feat1_processed, feat2_processed

    def warp_multi_channel_features(self, x_features, rb_features, g_features, flows_forward, flows_backward, center_idx, 
                                   mask_over_x_list):
        """Align X/RB/G features across frames and apply mask-guided enhancement."""
        n = len(x_features)
        b, _, h, w = x_features[0].shape

        aligned_x_forward = []
        aligned_rb_forward = [] if self.RB_guidance else None
        aligned_g_forward = [] if self.G_guidance else None
        
        feat_prop_x = x_features[0].new_zeros(b, self.num_feat, h, w)
        feat_prop_rb = feat_prop_x.clone() if self.RB_guidance else None
        feat_prop_g = feat_prop_x.clone() if self.G_guidance else None
        
        # 分别维护各通道的传播掩码
        propagated_mask_x = None
        
        for i in range(center_idx + 1):
            curr_mask_x = mask_over_x_list[i]
            
            if i == 0:
                feat_prop_x = x_features[i]
                propagated_mask_x = curr_mask_x
                
                if self.RB_guidance:
                    feat_prop_rb = rb_features[i]
                    
                if self.G_guidance:
                    feat_prop_g = g_features[i]

            else:
                flow = flows_forward[:, i - 1, :, :, :]
                flow_permuted = flow.permute(0, 2, 3, 1)

                feat_prop_x = flow_warp(feat_prop_x, flow_permuted)
                propagated_mask_x = flow_warp(
                    propagated_mask_x.unsqueeze(1), 
                    flow_permuted, 
                    interp_mode='nearest', 
                    padding_mode='zeros'
                ).squeeze(1)

                curr_x_feat = x_features[i]
                curr_x_enhanced = curr_x_feat
                feat_prop_x_enhanced = feat_prop_x
                if self.mask_guide:
                    feat_prop_x_enhanced, curr_x_enhanced = self.complementary_feature_enhancement(
                        feat_prop_x, curr_x_feat, propagated_mask_x, curr_mask_x
                    )
                feat_prop_x = self.attn_merge(curr_x_enhanced, feat_prop_x_enhanced)
                propagated_mask_x = torch.max(curr_mask_x, propagated_mask_x)

                if self.RB_guidance:
                    feat_prop_rb = flow_warp(feat_prop_rb, flow_permuted)                  
                    curr_rb_feat = rb_features[i]
                    feat_prop_rb = self.attn_merge(curr_rb_feat, feat_prop_rb)

                if self.G_guidance:
                    feat_prop_g = flow_warp(feat_prop_g, flow_permuted)                    
                    curr_g_feat = g_features[i]
                    feat_prop_g = self.attn_merge(curr_g_feat, feat_prop_g)

            aligned_x_forward.append(feat_prop_x.clone())
            if self.RB_guidance and aligned_rb_forward is not None:
                aligned_rb_forward.append(feat_prop_rb.clone())
            if self.G_guidance and aligned_g_forward is not None:
                aligned_g_forward.append(feat_prop_g.clone())

        aligned_x_backward = [None] * n
        aligned_rb_backward = [None] * n if self.RB_guidance else None
        aligned_g_backward = [None] * n if self.G_guidance else None
        
        feat_prop_x = x_features[-1].new_zeros(b, self.num_feat, h, w)
        feat_prop_rb = feat_prop_x.clone() if self.RB_guidance else None
        feat_prop_g = feat_prop_x.clone() if self.G_guidance else None
        
        propagated_mask_x = None
        
        for i in range(n - 1, center_idx - 1, -1):
            curr_mask_x = mask_over_x_list[i]
            
            if i == n - 1:
                feat_prop_x = x_features[i]
                propagated_mask_x = curr_mask_x
                
                if self.RB_guidance:
                    feat_prop_rb = rb_features[i]
                    
                if self.G_guidance:
                    feat_prop_g = g_features[i]

            else:
                flow = flows_backward[:, i, :, :, :]
                flow_permuted = flow.permute(0, 2, 3, 1)

                feat_prop_x = flow_warp(feat_prop_x, flow_permuted)
                propagated_mask_x = flow_warp(
                    propagated_mask_x.unsqueeze(1),
                    flow_permuted,
                    interp_mode='nearest',
                    padding_mode='zeros'
                ).squeeze(1)

                curr_x_feat = x_features[i]
                curr_x_enhanced = curr_x_feat
                feat_prop_x_enhanced = feat_prop_x
                if self.mask_guide:
                    feat_prop_x_enhanced, curr_x_enhanced = self.complementary_feature_enhancement(
                        feat_prop_x, curr_x_feat, propagated_mask_x, curr_mask_x
                    )
                feat_prop_x = self.attn_merge(curr_x_enhanced, feat_prop_x_enhanced)
                propagated_mask_x = torch.max(curr_mask_x, propagated_mask_x)

                if self.RB_guidance:
                    feat_prop_rb = flow_warp(feat_prop_rb, flow_permuted)
                    curr_rb_feat = rb_features[i]
                    feat_prop_rb = self.attn_merge(curr_rb_feat, feat_prop_rb)

                if self.G_guidance:
                    feat_prop_g = flow_warp(feat_prop_g, flow_permuted)
                    curr_g_feat = g_features[i]
                    feat_prop_g = self.attn_merge(curr_g_feat, feat_prop_g)

            aligned_x_backward[i] = feat_prop_x.clone()
            if self.RB_guidance and aligned_rb_backward is not None:
                aligned_rb_backward[i] = feat_prop_rb.clone()
            if self.G_guidance and aligned_g_backward is not None:
                aligned_g_backward[i] = feat_prop_g.clone()
        
        return aligned_x_forward, aligned_rb_forward, aligned_g_forward, aligned_x_backward, aligned_rb_backward, aligned_g_backward

    def forward(self, x, wb, cam2rgb):
        """Forward function with complementary mask-based feature enhancement."""
        b, n, _, h, w = x.size()

        mask_over_x_list = []
        mask_under_x_list = []
        
        for i in range(n):
            x_i = x[:, i, :, :, :]
            mask_over_x_i, mask_under_x_i = self.compute_channel_specific_mask(x_i)
            mask_over_x_list.append(mask_over_x_i)
            mask_under_x_list.append(mask_under_x_i)

        out_frames = []
        for i in range(n):
            frame_bayer = x[:, i]
            wb_i = wb[:, i] if wb.ndim == 3 else wb
            cam2rgb_i = cam2rgb[:, i] if cam2rgb.ndim == 4 else cam2rgb
            proc = process(frame_bayer, wb_i, cam2rgb_i, gamma=2.2, use_demosaic=False, use_tonemapping=False, data_range=1.0)
            out_frames.append(proc)
        lqs_rgb = torch.stack(out_frames, dim=1)

        flows_forward, flows_backward = self.get_flow_bidirectional(lqs_rgb)

        center_idx = n // 2

        x_features = []
        rb_features = [] if self.RB_guidance else None
        g_features = [] if self.G_guidance else None
        
        for i in range(n):
            x_i = x[:, i, :, :, :]

            x_feat = self.shallow_extraction(x_i)
            x_features.append(x_feat)

            if self.RB_guidance:
                rb_i = x_i[:, [0, 2], :, :]
                rb_feat = self.rb_feature_extractor(rb_i)
                rb_features.append(rb_feat)

            if self.G_guidance:
                g_i = x_i[:, [1, 3], :, :]
                g_feat = self.g_feature_extractor(g_i)
                g_features.append(g_feat)

        (aligned_x_forward, aligned_rb_forward, aligned_g_forward, 
         aligned_x_backward, aligned_rb_backward, aligned_g_backward) = self.warp_multi_channel_features(
            x_features, rb_features, g_features, flows_forward, flows_backward, center_idx, 
            mask_over_x_list
        )

        feat_forward_x = aligned_x_forward[center_idx]
        feat_backward_x = aligned_x_backward[center_idx]

        mask_over_center = mask_over_x_list[center_idx]
        mask_under_center = mask_under_x_list[center_idx]
        mask_normal_center = 1.0 - mask_over_center - mask_under_center

        mask_over_expand = mask_over_center.unsqueeze(1)
        mask_under_expand = mask_under_center.unsqueeze(1)
        mask_normal_expand = mask_normal_center.unsqueeze(1)

        features_to_fuse = [feat_forward_x, feat_backward_x]

        if self.RB_guidance and aligned_rb_forward is not None:
            feat_forward_rb = aligned_rb_forward[center_idx]
            feat_backward_rb = aligned_rb_backward[center_idx]

            rb_weight = mask_over_expand * 2.0 + mask_normal_expand * 1.0 + mask_under_expand * 0.3
            feat_forward_rb_weighted = feat_forward_rb * rb_weight
            feat_backward_rb_weighted = feat_backward_rb * rb_weight
            
            features_to_fuse.extend([feat_forward_rb_weighted, feat_backward_rb_weighted])

        if self.G_guidance and aligned_g_forward is not None:
            feat_forward_g = aligned_g_forward[center_idx]
            feat_backward_g = aligned_g_backward[center_idx]

            g_weight = mask_under_expand * 2.0 + mask_normal_expand * 1.0 + mask_over_expand * 0.3
            feat_forward_g_weighted = feat_forward_g * g_weight
            feat_backward_g_weighted = feat_backward_g * g_weight
            
            features_to_fuse.extend([feat_forward_g_weighted, feat_backward_g_weighted])

        fused_multichannel_feat = torch.cat(features_to_fuse, dim=1)
        feat_prop = self.multi_channel_fusion(fused_multichannel_feat)

        out_enc_level1 = self.encoder_level1(feat_prop)
        
        x_center = x[:, center_idx, :, :, :]
        inp_enc_level2 = self.down1_2(out_enc_level1)
        x_center_inl2 = self.l2(x_center)
        out_enc_level2 = self.encoder_level2(inp_enc_level2 + x_center_inl2)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)
        x_center_inl3 = self.l3(x_center)
        latent = self.latent(inp_enc_level3 + x_center_inl3)
        
        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.reduce_chan_level1(out_dec_level1)
        
        out_1 = self.refinement(out_dec_level1 + feat_prop)

        out = self.lrelu(self.conv_hr(out_1))
        out = self.conv_last(out)

        x_center_out = x[:, center_idx, :, :, :]
        out += x_center_out
        
        out_l = out

        mask_under_center = mask_under_x_list[center_idx]
        mask_over_center = mask_over_x_list[center_idx]
        
        return out_l, mask_under_center.unsqueeze(1), mask_over_center.unsqueeze(1)
    

class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)

