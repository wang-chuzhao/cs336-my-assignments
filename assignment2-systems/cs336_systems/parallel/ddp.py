import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp

class DDP(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module=module
        self.handles=[]

        self.world_size=dist.get_world_size()

        if self.world_size>1:
            #收到权重tensor的rank原地修改
            for param in self.module.parameters():
                dist.broadcast(param.data,src=0)

            for param in self.module.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self,param):
        param.grad.div_(self.world_size)
        handle=dist.all_reduce(param.grad.data,op=dist.ReduceOp.SUM,async_op=True)
        self.handles.append(handle)
                    
    
    def forward(self, *args, **kwargs):
        self.handles.clear()
        return self.module(*args,**kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()