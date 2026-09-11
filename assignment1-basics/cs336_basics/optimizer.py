import torch
import math
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Iterable

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it<warmup_iters:
        return it/warmup_iters*max_learning_rate
    elif it<=cosine_cycle_iters:
        return min_learning_rate+0.5*(1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], 
    max_l2_norm: float,
    eps:float=1e-6,
) -> None:
    parameters=list(parameters)
    g_norm=0
    for p in parameters:
        if p.grad is not None:
            g_norm+=(p.grad**2).sum()
    g_norm=g_norm.sqrt()
    if g_norm>max_l2_norm:
        scale=max_l2_norm/(g_norm+eps)
        for p in parameters:
            if p.grad is not None:
                p.grad*=scale

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        parameters,
        lr,
        betas,
        eps,
        weight_decay,
    ):
        defaults={
            'lr':lr,
            'betas':betas,
            'eps':eps,
            'weight_decay':weight_decay
        }

        super().__init__(parameters,defaults)

    def step(self, closure = None):
        for group in self.param_groups:
            lr=group['lr']
            beta1,beta2=group['betas']
            eps=group['eps']
            weight_decay=group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad=p.grad.data
                if not torch.isfinite(grad).all():
                    continue

                state=self.state[p]
                if len(state)==0:
                    state['step']=0
                    state['m']=torch.zeros_like(p.data)
                    state['v']=torch.zeros_like(p.data)

                state['step']+=1
                t=state['step']
                m,v=state['m'],state['v']
                grad=p.grad.data

                lr_t=lr*math.sqrt(1-beta2**t)/(1-beta1**t)
                m.mul_(beta1).add_(grad,alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad,grad,value=1-beta2)

                p.data.mul_(1-lr*weight_decay)
                denom=v.sqrt().add_(eps)
                p.data.addcdiv_(m,denom,value=-lr_t)