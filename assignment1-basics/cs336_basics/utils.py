import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import numpy as np
import numpy.typing as npt
import os
from typing import BinaryIO, IO
from .tokenizer import Tokenizer

def softmax(
    in_features:Float[Tensor,"..."],
    dim:int
)->Float[Tensor,"..."]:
    #amax沿指定维度求最大值，返回Tensor;max返回namedtuple
    x_max=in_features.amax(dim=dim,keepdim=True)
    x_exp=(in_features-x_max).exp()
    scale=x_exp.sum(dim=dim,keepdim=True)
    return x_exp/scale

def cross_entropy(
    inputs:Float[Tensor,"batch_size vocab_size"],
    targets:Int[Tensor,"batch_size"]
)->Float[Tensor,""]:
    #高级索引，idx_target和target都表示inputs的索引
    idx_targets=torch.arange(inputs.size(0))
    y=inputs[idx_targets,targets]
    x_max=inputs.amax(dim=-1,keepdim=True)
    log_sum_exp=(inputs-x_max).exp().sum(dim=-1).log()
    return (log_sum_exp-y+x_max).mean()

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts=np.random.randint(
        low=0,
        high=len(dataset)-context_length,
        size=batch_size
    )

    #fancy indexing, get index(batch_size,context_length)
    index=starts[:,None]+np.arange(context_length)[None,:]
    inputs=dataset[index]
    targets=dataset[index+1]

    inputs=torch.from_numpy(inputs).long().to(device)
    targets=torch.from_numpy(targets).long().to(device)

    return (inputs,targets)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    dict={
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'iteration':iteration,
    }
    torch.save(dict,out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    dict=torch.load(src)
    model.load_state_dict(dict['model_state_dict'])
    optimizer.load_state_dict(dict['optimizer_state_dict'])
    return dict['iteration']

def txt_to_bin(
    tokenizer:Tokenizer,
    txt_path:str,
    bin_path:str,
    dtype=np.uint16,
    chunk_size=500_000,
    log_every_chunks=20,
):
    buffer=[]
    chunk_count=0
    total_tokens=0
    print(f"Encoding {txt_path} -> {bin_path}", flush=True)
    with open(txt_path,encoding="utf-8") as f_in,open(bin_path,"wb") as f_out:
        for token_id in tokenizer.encode_iterable(f_in):
            buffer.append(token_id)
            if len(buffer)>=chunk_size:
                np.array(buffer,dtype=dtype).tofile(f_out)
                chunk_count+=1
                total_tokens+=chunk_size
                buffer.clear()
                if chunk_count%log_every_chunks==0:
                    size_mb=os.path.getsize(bin_path)/(1024*1024)
                    print(
                        f"  {os.path.basename(bin_path)}: {total_tokens:,} tokens, {size_mb:.1f} MB",
                        flush=True,
                    )
        if buffer:
            np.array(buffer,dtype=dtype).tofile(f_out)
            total_tokens+=len(buffer)
    size_mb=os.path.getsize(bin_path)/(1024*1024)
    print(
        f"Done {os.path.basename(bin_path)}: {total_tokens:,} tokens, {size_mb:.1f} MB",
        flush=True,
    )