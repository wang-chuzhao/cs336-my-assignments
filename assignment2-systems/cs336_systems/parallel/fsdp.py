import torch
import torch.nn as nn
import torch.distributed as dist
from cs336_basics.model import Embedding,Linear

class FSDP(nn.Module):
    #该模型要实现param\data\grad\optimizer四种数据的切分
    def __init__(self,module:nn.Module,compute_dtype:torch.dtype | None=None):
        super().__init__()
        self.module=module
        self.rank=dist.get_rank()
        self.world_size=dist.get_world_size()
        self.compute_dtype=compute_dtype
        self.mod_params=[]
        self.grad_handles=[]
        self.origin_dtype=None
        self.fp32_shard={}

        self.units=[] # [emb, lin1, lin2, ...]
        self.units_ready=False
        self.inflights={} # id(module) -> (handle, full)

        hooked=set()

        for name,mod in self.module.named_modules():
            if isinstance(mod,(Embedding,Linear)):
                #将权重分片保存
                #mod.parameters()中只有weight
                p=mod.weight
                if self.origin_dtype is None:
                    self.origin_dtype=p.dtype
                if p.requires_grad:
                    full_length=p.shape[0]
                    part_length=full_length//self.world_size
                    start=self.rank*part_length
                    end=start+part_length
                    new_weight=p.data[start:end,:]
                    #if end-start<full_length:
                    #    new_weight.pad()
                    weight_param={
                        'name':name+".weight",
                        'full_shape':p.shape,
                        'param':p
                    }
                    self.mod_params.append(weight_param)
                    p.data=new_weight.clone()
                    mod.register_forward_pre_hook(self.fwd_pre)
                    mod.register_forward_hook(self.fwd_post)
                    if isinstance(mod,Linear):
                        mod.register_full_backward_pre_hook(self.bwd_pre)

                    if id(p) not in hooked:
                        p.register_post_accumulate_grad_hook(self.sharded_grad_hook)
                        hooked.add(id(p))

        for mod in self.module.modules():
            for p in mod.parameters(recurse=False):
                if p.requires_grad and id(p) not in hooked:
                    p.register_post_accumulate_grad_hook(self.full_grad_hook)

    def _next_unit(self,module):
        i=self.units.index(module)
        return self.units[i+1] if i+1<len(self.units) else None

    def _start_unshard(self,module):
        if id(module) in self.inflights:
            return
        shard=module.weight.data
        self.fp32_shard[id(module.weight)]=shard.clone()
        if self.compute_dtype is not None:
            shard=shard.to(self.compute_dtype)
        shape=shard.shape
        full_length=shape[0]*self.world_size
        output=torch.empty((full_length,shape[1]),dtype=shard.dtype,device=shard.device)
        handle=dist.all_gather_into_tensor(output,shard.contiguous(),async_op=True)
        self.inflights[id(module)]=(handle,output)

    def fwd_pre(self,module,*args):
        if not self.units_ready:
            self.units.append(module)

        if id(module) not in self.inflights:#第一层
            self._start_unshard(module)
        handle,output=self.inflights.pop(id(module))
        handle.wait()
        module.weight.data=output

        if self.units_ready:
            next_unit=self._next_unit(module)
            if next_unit is not None:
                self._start_unshard(next_unit)

    def fwd_post(self,module,*args):
        module.weight.data=self.fp32_shard[id(module.weight)]

    def _pre_unit(self,module):
        i=self.units.index(module)
        return self.units[i-1] if i-1>=0 else None

    def bwd_pre(self,module,*args):
        if id(module) not in self.inflights:
            self._start_unshard(module)
        handle,output=self.inflights.pop(id(module))
        handle.wait()
        module.weight.data=output
        prev_unit=self._pre_unit(module)
        if isinstance(prev_unit,Linear):
            self._start_unshard(prev_unit)

    def sharded_grad_hook(self,param):
        #此时Linear是bwd_pre后的full_shape，但是Embedding仍然是分片的
        full_shape=None
        for weight_param in self.mod_params:
            if weight_param['param'] is param:
                full_shape=weight_param['full_shape']
                break
        part_length=full_shape[0]//self.world_size
        #先处理Linear的分片
        if param.data.shape[0]==full_shape[0]:
            param.data=self.fp32_shard.pop(id(param))

        #两种模块的grad都是full_shape，因为grad的shape和fwd时记录的weight的shape是一致的
        full_grad=param.grad.data.to(self.origin_dtype)/self.world_size
        output=torch.empty((part_length,full_shape[1]),dtype=full_grad.dtype,device=full_grad.device)
        handle=dist.reduce_scatter_tensor(output,full_grad.contiguous(),op=dist.ReduceOp.SUM,async_op=True)
        self.grad_handles.append(handle)
        #最终操作的是output的内存，异步操作不影响最终结果
        param.grad=output

    def full_grad_hook(self,param):
        param.grad.data/=self.world_size
        handle=dist.all_reduce(param.grad.data,op=dist.ReduceOp.SUM,async_op=True)
        self.grad_handles.append(handle)

    def forward(self,*args,**kwargs):
        output=self.module(*args,**kwargs)
        if not self.units_ready:
            self.units_ready=True
        return output

    def finish_gradient_synchronization(self):
        for handle in self.grad_handles:
            handle.wait()
        self.grad_handles.clear()

    def gather_full_params(self):
        results={}
        sharded=set()
        for weight_params in self.mod_params:
            name=weight_params['name']
            full_shape=weight_params['full_shape']
            param=weight_params['param']
            output=torch.empty(full_shape,dtype=param.data.dtype,device=param.data.device)
            dist.all_gather_into_tensor(output,param.data)
            sharded.add(id(param))
            results[name]=output

        for name,p in self.module.named_parameters():
            if id(p) not in sharded:
                results[name]=p.data

        return results