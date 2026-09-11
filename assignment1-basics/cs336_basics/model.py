import torch
import torch.nn as nn
from jaxtyping import Float, Int, Bool
from torch import Tensor
import cs336_basics.utils as utils

def silu(
    in_features:Float[Tensor,"..."]
)->Float[Tensor,"..."]:
        return in_features/(1+torch.exp(-in_features))

class Linear(nn.Module):
    def __init__(
        self,
        d_in:int,  
        d_out:int,
        weights:Float[Tensor,"d_out d_in"]|None=None,
    ):
        super().__init__()
        if weights is not None:
            self.weight=weights
        else:
            self.weight=nn.Parameter(torch.randn((d_out,d_in))*0.02)
    
    def forward(
        self,
        in_features:Float[Tensor,"... d_in"]
    ):
        return in_features@self.weight.transpose(-1,-2)
    
class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size:int,
        d_model:int,
        weights:Float[Tensor,"vocab_size d_model"]|None=None,
    ):
        super().__init__()
        if weights is not None:
            self.weight=weights
        else:
            self.weight=nn.Parameter(torch.randn((vocab_size,d_model))*0.02)

    def forward(
        self,
        token_ids:Int[Tensor,"..."]
    ):
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model:int,
        eps:float,
        weights:Float[Tensor,"d_model"]|None=None,
    ):
        super().__init__()
        self.eps=eps
        if weights is None:
            self.weights=nn.Parameter(torch.ones(d_model))
        else:
            self.weights=weights

    def forward(
        self,
        in_features:Float[Tensor,"... d_model"]
    )->Float[Tensor,"... d_model"]:
        in_dtype=in_features.dtype
        in_features=in_features.to(torch.float32)
        scale=torch.sqrt((in_features**2).mean(dim=-1,keepdim=True)+self.eps)
        return (in_features/scale*self.weights).to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        w1_weight: Float[Tensor, " d_ff d_model"]|None=None,
        w2_weight: Float[Tensor, " d_model d_ff"]|None=None,
        w3_weight: Float[Tensor, " d_ff d_model"]|None=None,
    ):
        super().__init__()
        self.w1=Linear(d_model,d_ff,w1_weight)
        self.w2=Linear(d_ff,d_model,w2_weight)
        self.w3=Linear(d_model,d_ff,w3_weight)

    def forward(
        self,
        in_features:Float[Tensor,"... d_model"]
    )->Float[Tensor,"... d_model"]:
        return self.w2(silu(self.w1(in_features))*self.w3(in_features))

class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
    ):
        super().__init__()
        theta_freqs=1.0/(theta**(torch.arange(0,d_k,2).float()/d_k))
        pos_freqs=torch.arange(max_seq_len).float()
        freqs=torch.outer(pos_freqs,theta_freqs)

        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(
        self,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        x_even=in_query_or_key[...,0::2]
        x_odd=in_query_or_key[...,1::2]
        cos=self.cos[token_positions]
        sin=self.sin[token_positions]
        y_even=x_even*cos-x_odd*sin
        y_odd=x_even*sin+x_odd*cos
        return torch.stack([y_even,y_odd],dim=-1).flatten(start_dim=-2)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k=Q.shape[-1]
    QKT=Q@K.transpose(-1,-2)/(d_k**(0.5))
    if mask is not None:
        QKT=QKT.masked_fill(~mask,float("-inf"))
    return utils.softmax(QKT,dim=-1)@V

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_model d_model"]|None=None,
        k_proj_weight: Float[Tensor, " d_model d_model"]|None=None,
        v_proj_weight: Float[Tensor, " d_model d_model"]|None=None,
        o_proj_weight: Float[Tensor, " d_model d_model"]|None=None,
    ):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_head=d_model//num_heads
        self.q_proj=Linear(d_model,d_model,q_proj_weight)
        self.k_proj=Linear(d_model,d_model,k_proj_weight)
        self.v_proj=Linear(d_model,d_model,v_proj_weight)
        self.o_proj=Linear(d_model,d_model,o_proj_weight)

    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_model"],
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        seq_len=in_features.shape[-2]
        Q=self.q_proj(in_features).unflatten(-1,(self.num_heads,self.d_head)).transpose(-2,-3)
        K=self.k_proj(in_features).unflatten(-1,(self.num_heads,self.d_head)).transpose(-2,-3)
        V=self.v_proj(in_features).unflatten(-1,(self.num_heads,self.d_head)).transpose(-2,-3)

        causal_mask=torch.tril(torch.ones(seq_len,seq_len,device=in_features.device)).bool()

        attn_output=scaled_dot_product_attention(Q,K,V,causal_mask)
        return self.o_proj(attn_output.transpose(-2,-3).flatten(start_dim=-2))
 
class MultiHeadSelfAttentionWithRope(MultiHeadSelfAttention):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        q_proj_weight: Float[Tensor, " d_model d_model"]|None=None,
        k_proj_weight: Float[Tensor, " d_model d_model"]|None=None,
        v_proj_weight: Float[Tensor, " d_model d_model"]|None=None,
        o_proj_weight: Float[Tensor, " d_model d_model"]|None=None,
    ):
        super().__init__(d_model,num_heads,q_proj_weight,k_proj_weight,v_proj_weight,o_proj_weight)
        self.RoPE=RotaryPositionEmbedding(self.d_head,theta,max_seq_len)
        
    def forward(
        self,
        in_features:Float[Tensor,"... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    )->Float[Tensor,"... sequence_length d_model"]:
        Q=self.q_proj(in_features).unflatten(-1,(self.num_heads,self.d_head)).transpose(-2,-3)
        K=self.k_proj(in_features).unflatten(-1,(self.num_heads,self.d_head)).transpose(-2,-3)
        V=self.v_proj(in_features).unflatten(-1,(self.num_heads,self.d_head)).transpose(-2,-3)

        seq_len=Q.shape[-2]
        if token_positions is None:
            token_positions=torch.arange(seq_len,device=in_features.device)
        
        Q_pos=self.RoPE(Q,token_positions)
        K_pos=self.RoPE(K,token_positions)

        causal_mask=torch.tril(torch.ones(seq_len,seq_len,device=in_features.device)).bool()

        attn_output=scaled_dot_product_attention(Q_pos,K_pos,V,causal_mask)

        return self.o_proj(attn_output.transpose(-2,-3).flatten(start_dim=-2))

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor]|None=None,
        eps:float=1e-5,
    ):
        super().__init__()
        if weights is None:
            weights={}
        self.ln1=RMSNorm(d_model,eps,weights.get("ln1.weight"))
        self.attn=MultiHeadSelfAttentionWithRope(
            d_model,
            num_heads,
            max_seq_len,
            theta,
            weights.get("attn.q_proj.weight"),
            weights.get("attn.k_proj.weight"),
            weights.get("attn.v_proj.weight"),
            weights.get("attn.output_proj.weight")
        )
        self.ln2=RMSNorm(d_model,eps,weights.get("ln2.weight"))
        self.ffn=SwiGLU(
            d_model,
            d_ff,
            weights.get("ffn.w1.weight"),
            weights.get("ffn.w2.weight"),
            weights.get("ffn.w3.weight")
        )

    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        x=in_features+self.attn(self.ln1(in_features))
        return x+self.ffn(self.ln2(x))

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor]|None=None,
        eps:float=1e-5,
    ):
        super().__init__()
        if weights is None:
            weights={}
        self.embd=Embedding(vocab_size,d_model,weights.get("token_embeddings.weight"))
        layer_weights_list=[]
        self.num_layers=num_layers
        for i in range(num_layers):
            prefix=f"layers.{i}."
            weights_dict={
                key.replace(prefix,""):value 
                for key,value in weights.items()
                if key.startswith(prefix)
            }
            layer_weights_list.append(weights_dict)
        self.tranformer=nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                rope_theta,
                layer_weights_list[i],
                eps,
            )
            for i in range(num_layers)
        ])
        self.ln_final=RMSNorm(d_model,eps,weights.get("ln_final.weight"))
        self.lm_head=Linear(d_model,vocab_size,weights.get("lm_head.weight"))

    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        in_features=self.embd(in_indices)
        for i in range(self.num_layers):
            in_features=self.tranformer[i](in_features)
        return self.lm_head(self.ln_final(in_features))