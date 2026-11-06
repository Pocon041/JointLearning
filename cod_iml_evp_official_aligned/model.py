"""EVP MiT-B4 with genuinely shared trainable prompts and two private decoders."""
from __future__ import annotations

import copy
import hashlib
import re
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .vendor_evp import Block,OverlapPatchEmbed,PromptGenerator
from .operators import FACA,ObjectConsistency


def convert_mmseg_mit(state):
    """Invert OpenMMLab's published MiT conversion, with no discarded tensors."""
    converted={}
    for key,value in state.items():
        match=re.fullmatch(r'layers\.(\d+)\.0\.(projection|norm)\.(weight|bias)',key)
        if match:
            stage,part,param=match.groups()
            converted[f'patch_embed{int(stage)+1}.'+('proj' if part=='projection' else 'norm')+'.'+param]=value
            continue
        match=re.fullmatch(r'layers\.(\d+)\.2\.(weight|bias)',key)
        if match:
            stage,param=match.groups();converted[f'norm{int(stage)+1}.{param}']=value;continue
        match=re.fullmatch(r'layers\.(\d+)\.1\.(\d+)\.(.+)',key)
        if not match:raise ValueError('Unmapped MiT pretrained key: '+key)
        stage,index,suffix=match.groups();prefix=f'block{int(stage)+1}.{index}.'
        if suffix.startswith('attn.attn.in_proj_'):
            param=suffix.rsplit('_',1)[-1];c=value.shape[0]//3
            converted[prefix+'attn.q.'+param]=value[:c]
            converted[prefix+'attn.kv.'+param]=value[c:]
        elif suffix.startswith('attn.attn.out_proj.'):
            converted[prefix+suffix.replace('attn.attn.out_proj.','attn.proj.')]=value
        elif suffix.startswith('ffn.layers.'):
            part,param=suffix[len('ffn.layers.'):].split('.')
            target={'0':'mlp.fc1','1':'mlp.dwconv.dwconv','4':'mlp.fc2'}.get(part)
            if target is None:raise ValueError('Unmapped MiT FFN key: '+key)
            if part in ('0','4') and param=='weight':value=value.squeeze(-1).squeeze(-1)
            converted[prefix+target+'.'+param]=value
        elif suffix.startswith(('norm1.','norm2.','attn.sr.','attn.norm.')):
            converted[prefix+suffix]=value
        else:raise ValueError('Unmapped MiT block key: '+key)
    return converted


class MiTBackbone(nn.Module):
    def __init__(self,channels,depths,heads,sr_ratios,drop_path_rate=.1,reference_img_size=352):
        super().__init__()
        self.channels,self.depths=tuple(channels),tuple(depths)
        dpr=torch.linspace(0,drop_path_rate,sum(depths)).tolist();offset=0
        norm=partial(nn.LayerNorm,eps=1e-6)
        for s,(width,depth,head,sr) in enumerate(zip(channels,depths,heads,sr_ratios),1):
            stage_img_size=reference_img_size if s==1 else reference_img_size//(2**s)
            setattr(self,f'patch_embed{s}',OverlapPatchEmbed(img_size=stage_img_size,
                patch_size=7 if s==1 else 3,stride=4 if s==1 else 2,
                in_chans=3 if s==1 else channels[s-2],embed_dim=width))
            setattr(self,f'block{s}',nn.ModuleList(Block(width,head,mlp_ratio=4,qkv_bias=True,
                norm_layer=norm,sr_ratio=sr,drop_path=dpr[offset+i]) for i in range(depth)))
            setattr(self,f'norm{s}',norm(width));offset+=depth

    def load_pretrained(self,path):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        saved=saved.get('state_dict',saved)
        clean={}
        for key,value in saved.items():
            for prefix in ['module.','backbone.']:
                if key.startswith(prefix):key=key[len(prefix):]
            clean[key]=value
        if clean and all(key.startswith('layers.') for key in clean):
            clean=convert_mmseg_mit(clean)
        expected=self.state_dict()
        if set(expected)<=set(clean):
            source={key:clean[key] for key in expected}
            ignored=sorted(set(clean)-set(expected))
            if any(not key.startswith('head.') for key in ignored):
                raise ValueError('Unexpected pretrained non-classifier keys: '+str(ignored[:8]))
            self.load_state_dict(source,strict=True)
            return {'loaded_keys':len(source),'ignored_keys':ignored,'source':'generic ImageNet1K MiT-B4'}
        raise ValueError('Pretrained key format needs explicit conversion; sample='+str(list(clean)[:8]))


class EVPPromptEncoder(nn.Module):
    def __init__(self,channels,depths,heads,sr_ratios,scale_factor,enhancement,
                 freeze_backbone,checkpoint_blocks,relation_depth,prototypes,drop_path_rate,reference_img_size=352):
        super().__init__()
        self.backbone=MiTBackbone(channels,depths,heads,sr_ratios,drop_path_rate,reference_img_size)
        self.prompt_generator=PromptGenerator(scale_factor,'highpass',channels,'1234',depths,
                                             'fft',.25,True,True,'adaptor',reference_img_size)
        self.task_tunes=nn.ModuleDict()
        tunes=[]
        for s,depth in enumerate(depths,1):
            stage=nn.ModuleList()
            for i in range(depth):
                name=f'lightweight_mlp{s}_{i}'
                stage.append(getattr(self.prompt_generator,name))
                delattr(self.prompt_generator,name)
            tunes.append(stage)
        self.task_tunes['cod']=nn.ModuleList(tunes)
        self.task_tunes['iml']=copy.deepcopy(self.task_tunes['cod'])
        self.enhancement=enhancement
        self.channels,self.depths=tuple(channels),tuple(depths)
        self.checkpoint_blocks=checkpoint_blocks
        self.freeze_backbone=freeze_backbone
        self.hfc_enabled=True
        if freeze_backbone:
            for p in self.backbone.parameters():p.requires_grad_(False)
        # Forks keep all original EVP tensors independent of optional modules.
        widths=[c//scale_factor for c in channels]
        if enhancement in ('structure','dual'):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed()+1001)
                self.structure=nn.ModuleList(FACA(c) for c in widths)
        if enhancement in ('trace','dual'):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed()+2001)
                self.trace=nn.ModuleList(ObjectConsistency(c,relation_depth,prototypes) for c in widths)
        if enhancement!='evp':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed()+3001)
                self.task_gates=nn.ModuleDict({t:nn.ModuleList(nn.Conv2d(4*c,2*c,1) for c in widths) for t in ['cod','iml']})
                for group in self.task_gates.values():
                    for layer in group:
                        nn.init.zeros_(layer.weight);nn.init.constant_(layer.bias,-2.)

    def forward(self,image,task):
        hfc=self.prompt_generator.init_handcrafted(image)
        if not self.hfc_enabled:hfc=tuple(torch.zeros_like(x) for x in hfc)
        x=image;output=[]
        for s in range(1,5):
            x,h,w=getattr(self.backbone,f'patch_embed{s}')(x)
            pe=getattr(self.prompt_generator,f'embedding_generator{s}')(x)
            base=pe+hfc[s-1]
            if self.enhancement!='evp':
                b,n,c=pe.shape
                content=pe.transpose(1,2).reshape(b,c,h,w)
                frequency=hfc[s-1].transpose(1,2).reshape(b,c,h,w)
                structure=self.structure[s-1](content+frequency) if hasattr(self,'structure') else frequency
                trace=self.trace[s-1](content,frequency) if hasattr(self,'trace') else frequency
                ds,dt=structure-frequency,trace-frequency
                gates=self.task_gates[task][s-1](torch.cat((content,frequency,ds,dt),dim=1)).sigmoid()
                gs,gt=gates.chunk(2,dim=1)
                correction=gs*ds+gt*dt
                base=base+correction.flatten(2).transpose(1,2)
            up=getattr(self.prompt_generator,f'shared_mlp{s}')
            for i,block in enumerate(getattr(self.backbone,f'block{s}')):
                prompt=up(self.task_tunes[task][s-1][i](base))
                if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                    # Explicit arguments avoid late-bound closure errors on recompute.
                    x=checkpoint(block,x+prompt,h,w,use_reentrant=False,preserve_rng_state=True)
                else:x=block(x+prompt,h,w)
            x=getattr(self.backbone,f'norm{s}')(x)
            x=x.transpose(1,2).reshape(image.shape[0],-1,h,w)
            output.append(x)
        return tuple(output)


class SegFormerHead(nn.Module):
    """Author decoder structure: four linear projections, concat, BN/ReLU, dropout."""
    def __init__(self,channels,width=768):
        super().__init__()
        self.projections=nn.ModuleList(nn.Linear(c,width) for c in channels)
        self.fuse=nn.Sequential(nn.Conv2d(4*width,width,1,bias=False),nn.BatchNorm2d(width),nn.ReLU(inplace=True))
        # MMCV ConvModule calls this MSRA initializer before linear_pred exists.
        nn.init.kaiming_normal_(self.fuse[0].weight,a=0,mode='fan_out',nonlinearity='relu')
        nn.init.ones_(self.fuse[1].weight)
        nn.init.zeros_(self.fuse[1].bias)
        self.dropout=nn.Dropout2d(.1)
        self.output=nn.Conv2d(width,1,1)

    def forward(self,features,size):
        maps=[]
        for x,projection in zip(features,self.projections):
            b,c,h,w=x.shape
            x=projection(x.flatten(2).transpose(1,2)).transpose(1,2).reshape(b,-1,h,w)
            maps.append(F.interpolate(x,size=features[0].shape[-2:],mode='bilinear',align_corners=False))
        # Original head concatenates c4,c3,c2,c1.
        logits=self.output(self.dropout(self.fuse(torch.cat(maps[::-1],dim=1))))
        return F.interpolate(logits,size=size,mode='bilinear',align_corners=False)


class JointEVPModel(nn.Module):
    def __init__(self,channels=(64,128,320,512),depths=(3,8,27,3),heads=(1,2,5,8),
                 sr_ratios=(8,4,2,1),decoder_width=768,scale_factor=4,enhancement='evp',
                 freeze_backbone=True,checkpoint_blocks=True,relation_depth=8,prototypes=16,
                 drop_path_rate=.1,hfc_enabled=True,reference_img_size=352):
        super().__init__()
        if enhancement not in ('evp','structure','trace','dual'):raise ValueError('Unknown enhancement')
        self.config=dict(channels=tuple(channels),depths=tuple(depths),heads=tuple(heads),sr_ratios=tuple(sr_ratios),
            decoder_width=decoder_width,scale_factor=scale_factor,enhancement=enhancement,
            freeze_backbone=freeze_backbone,checkpoint_blocks=checkpoint_blocks,
            relation_depth=relation_depth,prototypes=prototypes,drop_path_rate=drop_path_rate,hfc_enabled=hfc_enabled,
            reference_img_size=reference_img_size)
        self.encoder=EVPPromptEncoder(channels,depths,heads,sr_ratios,scale_factor,enhancement,
            freeze_backbone,checkpoint_blocks,relation_depth,prototypes,drop_path_rate,reference_img_size)
        self.encoder.hfc_enabled=hfc_enabled
        self.cod_decoder=SegFormerHead(channels,decoder_width)
        self.iml_decoder=copy.deepcopy(self.cod_decoder)
        self._pretrained_sha256=None

    def forward(self,x,task=None,*,return_aux=False):
        if task not in (None,'cod','iml'):raise ValueError('Unknown task')
        if x.ndim!=4 or x.shape[1]!=3 or not x.is_floating_point():raise ValueError('Expected float RGB')
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
        std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        image=(x-mean)/std
        outputs={}
        for t in ['cod','iml']:
            if task not in (None,t):continue
            features=self.encoder(image,t)
            outputs[t]=getattr(self,t+'_decoder')(features,x.shape[-2:])
        return outputs

    def load_spatial_pretrained(self,path,source='auto'):
        result=self.encoder.backbone.load_pretrained(path)
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        self._pretrained_sha256=h.hexdigest()
        return result

    def parameter_groups(self):
        cod=list(self.cod_decoder.parameters())+list(self.encoder.task_tunes['cod'].parameters())
        iml=list(self.iml_decoder.parameters())+list(self.encoder.task_tunes['iml'].parameters())
        if hasattr(self.encoder,'task_gates'):
            cod+=list(self.encoder.task_gates['cod'].parameters());iml+=list(self.encoder.task_gates['iml'].parameters())
        private={id(p) for p in cod+iml}
        shared=[p for p in self.parameters() if p.requires_grad and id(p) not in private]
        return {'shared':shared,'cod_private':cod,'iml_private':iml}

    def training_state(self):
        """Only omit immutable generic backbone tensors; keep all learned buffers."""
        if self.encoder.freeze_backbone and self._pretrained_sha256 is None:
            raise ValueError('Sparse checkpoints require a verified immutable pretrained file')
        return {k:v.detach().cpu() for k,v in self.state_dict().items()
                if not (self.encoder.freeze_backbone and k.startswith('encoder.backbone.'))}

    def restore_training_state(self,state,pretrained_sha256):
        if self.encoder.freeze_backbone and self._pretrained_sha256!=pretrained_sha256:
            raise ValueError('Immutable pretrained checkpoint hash differs')
        result=self.load_state_dict(state,strict=False)
        expected={k for k in self.state_dict() if self.encoder.freeze_backbone and k.startswith('encoder.backbone.')}
        if set(result.missing_keys)!=expected or result.unexpected_keys:
            raise ValueError('Sparse checkpoint misses learned tensors or contains unexpected tensors')
