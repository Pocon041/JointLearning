"""Matched EVP objectives and symmetric updates for unpaired COD/CASIA batches."""
from contextlib import nullcontext
import math
import torch
from torch.nn import functional as F
from cod_iml_joint.engine import _validate_batch
from cod_iml_joint.losses import validate_weight


def task_loss(task,logits,target):
    bce=F.binary_cross_entropy_with_logits(logits.float(),target.float())
    if task=='iml':return bce
    if task!='cod':raise ValueError('Unknown task')
    probability=logits.float().sigmoid();target=target.float()
    intersection=(probability*target).sum((2,3))
    union=(probability+target).sum((2,3))-intersection
    return bce+(1-intersection/union.clamp_min(1e-6)).mean()


def joint_train_step(model,optimizer,cod_batch=None,iml_batch=None,cod_weight=1.,iml_weight=1.,
                     *,boundary_weight=0.,amp=False,max_grad_norm=5.):
    if boundary_weight!=0:raise ValueError('EVP comparison has no added boundary loss')
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm<=0):raise ValueError('Invalid clipping norm')
    active=[(t,b,validate_weight(w,t+'_weight')) for t,b,w in
            [('cod',cod_batch,cod_weight),('iml',iml_batch,iml_weight)] if b is not None and w>0]
    if not active:raise ValueError('No active task')
    for t,b,w in active:_validate_batch(b,t)
    optimizer.zero_grad(set_to_none=True);model.train();logs={};total=None;previous=None
    shared=[p for g in optimizer.param_groups if g['name'].startswith('shared') for p in g['params']]
    def shared_vector():
        return torch.cat([(p.grad.detach().float() if p.grad is not None else torch.zeros_like(p,dtype=torch.float32)).flatten() for p in shared])
    try:
        for task,batch,weight in active:
            context=torch.autocast(batch['image'].device.type,dtype=torch.bfloat16) if amp else nullcontext()
            with context:
                output=model(batch['image'],task=task)[task]
                loss=task_loss(task,output,batch['mask']);weighted=weight*loss
            if not torch.isfinite(weighted):raise FloatingPointError('Nonfinite task loss')
            weighted.backward();logs[task]=loss.detach()
            total=weighted.detach() if total is None else total+weighted.detach()
            if len(active)==2 and shared:
                current=shared_vector()
                gradient=current if previous is None else current-previous
                logs['shared_grad_'+task]=gradient.norm()
                if previous is not None:
                    logs['shared_grad_cosine']=F.cosine_similarity(previous[None],gradient[None],eps=1e-12)[0]
                previous=current
        params=[p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        if not params or any(not torch.isfinite(p.grad).all() for p in params):raise FloatingPointError('Invalid gradients')
        if max_grad_norm is not None:logs['grad_norm']=torch.nn.utils.clip_grad_norm_(params,max_grad_norm,error_if_nonfinite=True).detach()
        optimizer.step()
    except BaseException:
        optimizer.zero_grad(set_to_none=True);raise
    logs['total']=total
    return logs
