"""Paper-derived FBNet FACA and ObjectFormer object/patch consistency units.

These implement the cited submodule equations, not complete original models.
The original EVP FFT/embedding/prompt path remains separately available.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveLowPass(nn.Module):
    """Predict a positive, unit-sum 3x3 kernel at each spatial position."""
    def __init__(self, channels):
        super().__init__()
        self.kernel = nn.Conv2d(channels, 9, 3, padding=1)

    def forward(self, x):
        b,c,h,w=x.shape
        weight=self.kernel(x).softmax(1)
        # Replication preserves constants; zero padding would create fake edges.
        patches=F.unfold(F.pad(x,(1,1,1,1),mode='replicate'),3).reshape(b,c,9,h,w)
        return (patches*weight[:,None]).sum(2)


class FACA(nn.Module):
    """FBNet equations 1-2: four adaptive LPs, dilated branches, six differences.

    Original n=4, kernel sizes=3, dilations=2/4/6/8. Output is projected back
    to prompt width. This is a paper reimplementation, not copied author code.
    """
    def __init__(self, channels):
        super().__init__()
        self.lowpass=nn.ModuleList(AdaptiveLowPass(channels) for _ in range(4))
        self.branches=nn.ModuleList(nn.Conv2d(channels,channels,3,padding=d,dilation=d,bias=False)
                                    for d in (2,4,6,8))
        self.aggregate=nn.Conv2d(6*channels,channels,1,bias=False)

    def forward(self,x):
        branches=[conv(lp(x)) for lp,conv in zip(self.lowpass,self.branches)]
        differences=[branches[i]-branches[j] for i in range(4) for j in range(i+1,4)]
        return self.aggregate(torch.cat(differences,dim=1))


class CrossAttention(nn.Module):
    """Separate normalized Q/K/V, as in ObjectFormer's released AnyAttention."""
    def __init__(self,width):
        super().__init__()
        self.norm_q,self.norm_k,self.norm_v=(nn.LayerNorm(width) for _ in range(3))
        self.q,self.k,self.v=(nn.Linear(width,width,bias=False) for _ in range(3))
        self.out=nn.Linear(width,width)

    def forward(self,q,k,v):
        q=self.q(self.norm_q(q));k=self.k(self.norm_k(k));v=self.v(self.norm_v(v))
        z=F.scaled_dot_product_attention(q[:,None],k[:,None],v[:,None],dropout_p=0.)
        return self.out(z[:,0])


class FeedForward(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.net=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,width))

    def forward(self,x):
        return self.net(self.norm(x))


def contextual_incoherence(x,kernel=3):
    """ObjectFormer equation 8: add mean local cosine similarity to each vector.

    Real neighbours only: replication padding avoids comparing padded zeros.
    This implements the paper BCIM rule, not the released code's extra local
    attention variant. It is NOT a guaranteed edit detector by itself.
    """
    b,c,h,w=x.shape
    normalized=F.normalize(x.float(),dim=1,eps=1e-6)
    local=F.unfold(F.pad(normalized,(kernel//2,)*4,mode='replicate'),kernel)
    local=local.reshape(b,c,kernel**2,h,w)
    similarity=(normalized[:,:,None]*local).sum(1).mean(1,keepdim=True)
    return x+similarity.to(x.dtype)


class ObjectPatchUpdate(nn.Module):
    """ObjectFormer object encoder, token-axis reasoning, patch decoder, BCIM."""
    def __init__(self,width,prototypes):
        super().__init__()
        self.object_attention=CrossAttention(width)
        self.object_norm=nn.LayerNorm(width)
        self.object_reasoning=nn.Conv1d(prototypes,prototypes,1,bias=False)
        self.object_ffn=FeedForward(width)
        self.patch_attention=CrossAttention(width)
        self.patch_ffn=FeedForward(width)

    def forward(self,objects,patches,h,w):
        objects=objects+self.object_attention(objects,patches,patches)
        objects=objects+self.object_reasoning(self.object_norm(objects))
        objects=objects+self.object_ffn(objects)
        patches=patches+self.patch_attention(patches,objects,objects)
        patches=patches+self.patch_ffn(patches)
        b,n,c=patches.shape
        # The two spatially corresponding modalities retain separate local grids.
        patches=patches.reshape(b,2,h*w,c).permute(0,1,3,2).reshape(b*2,c,h,w)
        patches=contextual_incoherence(patches)
        patches=patches.reshape(b,2,c,h*w).permute(0,1,3,2).reshape(b,n,c)
        return objects,patches


class ObjectConsistency(nn.Module):
    """Eight object/patch refinements and 16 prototypes from the paper settings.

    Input encoders are EVP's stage-aligned RGB embedding and HFC embedding.
    Thus this migrates relation processing, not ObjectFormer's entire HFE/stem.
    """
    def __init__(self,width,depth=8,prototypes=16):
        super().__init__()
        self.prototypes=nn.Parameter(torch.randn(1,prototypes,width)*.02)
        self.blocks=nn.ModuleList(ObjectPatchUpdate(width,prototypes) for _ in range(depth))
        self.project=nn.Conv2d(2*width,width,1)

    def forward(self,content,frequency):
        b,c,h,w=frequency.shape
        patches=torch.cat((content.flatten(2).transpose(1,2),frequency.flatten(2).transpose(1,2)),dim=1)
        objects=self.prototypes.expand(b,-1,-1)
        for block in self.blocks:
            objects,patches=block(objects,patches,h,w)
        maps=patches.reshape(b,2,h*w,c).permute(0,1,3,2).reshape(b,2*c,h,w)
        return self.project(maps)
