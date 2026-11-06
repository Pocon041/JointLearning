"""Explicit shared/private learning rates and standard warmup + cosine decay."""
import math
import torch


def build_optimizer(model,config):
    roles=model.parameter_groups()
    rates={'shared':float(config.get('shared_lr',2e-4)),
           'cod_private':float(config.get('cod_private_lr',2e-4)),
           'iml_private':float(config.get('iml_private_lr',5e-4))}
    if any(not math.isfinite(v) or v<=0 for v in rates.values()):raise ValueError('Learning rates must be positive')
    decay=float(config.get('weight_decay',.01))
    if decay<0 or not math.isfinite(decay):raise ValueError('Invalid weight decay')
    seen=set();groups=[];report=[]
    for role,parameters in roles.items():
        split={True:[],False:[]}
        for p in parameters:
            if not p.requires_grad:continue
            if id(p) in seen:raise ValueError('Overlapping optimizer roles')
            seen.add(id(p));use_decay=bool(config.get('decay_bias_and_norm',False)) or p.ndim>1
            split[use_decay].append(p)
        for decayed,params in split.items():
            if not params:continue
            name=('shared_new' if role=='shared' else role)+('_decay' if decayed else '_no_decay')
            wd=decay if decayed else 0.
            groups.append({'params':params,'lr':rates[role],'weight_decay':wd,'name':name})
            report.append({'name':name,'parameters':sum(p.numel() for p in params),'lr':rates[role],'weight_decay':wd})
    expected={id(p) for p in model.parameters() if p.requires_grad}
    if seen!=expected:raise ValueError('Optimizer does not cover exactly all trainable parameters')
    return torch.optim.AdamW(groups,betas=tuple(config.get('betas',(.9,.999)))),report


def make_scheduler(optimizer,total_steps,warmup_steps,min_lr=1e-6,start_factor=.1):
    if total_steps<1 or not 0<=warmup_steps<total_steps:raise ValueError('Invalid scheduler duration')
    if min_lr<0 or any(min_lr>g['lr'] for g in optimizer.param_groups):raise ValueError('Invalid eta_min')
    if not 0<start_factor<=1:raise ValueError('Invalid warmup start factor')
    cosine=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total_steps-warmup_steps,eta_min=min_lr)
    if warmup_steps==0:return cosine
    warmup=torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=start_factor,end_factor=1.,total_iters=warmup_steps)
    return torch.optim.lr_scheduler.SequentialLR(optimizer,[warmup,cosine],milestones=[warmup_steps])
