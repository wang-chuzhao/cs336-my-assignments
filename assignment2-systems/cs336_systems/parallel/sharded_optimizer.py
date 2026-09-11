import torch
import torch.distributed as dist
import torch.optim as optim
from typing import Type,Any

class ShardedOptimizer(optim.Optimizer):
    def __init__(self,params,optimizer_cls:Type[optim.Optimizer],**kwargs:Any):
        self.world_size=dist.get_world_size()
        self.rank=dist.get_rank()

        self.params_to_rank={}
        self.optimizer_cls=optimizer_cls
        self.local_optimizer=None
        self.param_counter=0
        #kwargs是dict
        self.optimizer_kwargs=kwargs

        #在parent类的__init__()中，会将params_group中的param_group逐个调用add_param_group()
        super().__init__(params,defaults=kwargs)

    def step(self,closure=None):
        if self.local_optimizer is None:
            loss=None
        else:
            loss=self.local_optimizer.step(closure)
        for param_group in self.param_groups:
            for p in param_group['params']:
                owner_rank=self.params_to_rank[id(p)]
                dist.broadcast(p.data,src=owner_rank)
        return loss

    def add_param_group(self,param_group:dict[str,Any]):
        #通过parant类实现generator的list化
        super().add_param_group(param_group)
        params=param_group['params']
        #if isinstance(params,list):
        #    params=list(params)

        new_params=[]
        for p in params:
            owner_rank=self.param_counter%self.world_size
            self.params_to_rank[id(p)]=owner_rank
            if owner_rank==self.rank:
                new_params.append(p)
            self.param_counter+=1

        if new_params:
            if self.local_optimizer is None:
                group_kwargs={k:v for k,v in param_group.items() if k!="params"}
                #后面的参数会覆盖前面的
                merged_kwargs={**self.optimizer_kwargs,**group_kwargs}
                self.local_optimizer=self.optimizer_cls(new_params,**merged_kwargs)
            else:
                new_group_kwargs={**param_group,'params':new_params}
                self.local_optimizer.add_param_group(new_group_kwargs)
