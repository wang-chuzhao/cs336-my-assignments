import triton
import torch
import triton.language as tl

@triton.jit
def attn_fwd_kernel(
    Q_ptr,K_ptr,V_ptr,O_ptr,L_ptr,
    #stride(0)表示在维度0上从i到i+1要跨过多少元素
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL:tl.constexpr,
):
    #program可以看作是CUDA block，则promgram_id可以理解为分块的起始点
    #program(i)表示grid第i维的编号，program(0)是分块的块编号，program(1)是batch的编号
    #取决于grid=(cdiv(n_queries,Q_TILE_SIZE),batch_size)
    query_tile_idx=tl.program_id(0)
    batch_idx=tl.program_id(1)

    #计算Q的三维索引
    query_indices=query_tile_idx*Q_TILE_SIZE+tl.arange(0,Q_TILE_SIZE)
    D_indices=tl.arange(0,D)
    q_offsets=batch_idx*stride_qb+query_indices[:,None]*stride_qq+D_indices[None,:]*stride_qd
    mask_q=(query_indices<N_QUERIES)[:,None]
    Query=tl.load(Q_ptr+q_offsets,mask=mask_q)

    Max=tl.full((Q_TILE_SIZE,),float("-inf"),dtype=tl.float32)
    Sum=tl.zeros((Q_TILE_SIZE,),dtype=tl.float32)
    Acc=tl.zeros((Q_TILE_SIZE,D),dtype=tl.float32)

    #使用tl内置函数来循环，可以直接编译成PTX
    for key_tile_idx in range(0,tl.cdiv(N_KEYS,K_TILE_SIZE)):
        key_indices=key_tile_idx*K_TILE_SIZE+tl.arange(0,K_TILE_SIZE)
        k_offsets=batch_idx*stride_kb+key_indices[:,None]*stride_kk+D_indices[None,:]*stride_kd
        mask_k=(key_indices<N_KEYS)[:,None]
        Key=tl.load(K_ptr+k_offsets,mask=mask_k)

        value_indices=key_indices
        value_offsets=batch_idx*stride_vb+value_indices[:,None]*stride_vk+D_indices[None,:]*stride_vd
        mask_v=mask_k
        Value=tl.load(V_ptr+value_offsets,mask=mask_v)

        scores=tl.dot(Query,tl.trans(Key))*scale

        if IS_CAUSAL:
            mask=query_indices[:,None]>=key_indices[None,:]
            scores=tl.where(mask,scores,float("-inf"))

        m_old=Max
        m_new=tl.max(scores,axis=1)
        #(n_queries,)
        Max=tl.maximum(m_old,m_new)
        alpha=tl.exp(m_old-Max)
        scores_new=tl.exp(scores-Max[:,None])
        Sum=alpha*Sum+tl.sum(scores_new,axis=1)
        Acc=alpha[:,None]*Acc+tl.dot(scores_new,Value)

    output=Acc/Sum[:,None]
    L=tl.log(Sum)+Max

    output_indices=query_indices
    output_offsets=batch_idx*stride_ob+output_indices[:,None]*stride_oq+D_indices[None,:]*stride_od
    tl.store(O_ptr+output_offsets,output,mask=mask_q)

    L_indices=query_indices
    L_offsets=batch_idx*stride_lb+L_indices*stride_lq
    mask_L=L_indices<N_QUERIES
    tl.store(L_ptr+L_offsets,L,mask=mask_L)

@triton.jit
def attn_bwd_kernel1(
    Q_ptr,K_ptr,V_ptr,O_ptr,L_ptr,
    dO_ptr,dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    dim: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL:tl.constexpr,
):
    query_tile_idx=tl.program_id(0)
    batch_idx=tl.program_id(1)

    q_indices=query_tile_idx*Q_TILE_SIZE+tl.arange(0,Q_TILE_SIZE)
    d_indices=tl.arange(0,dim)

    q_offsets=batch_idx*stride_qb+q_indices[:,None]*stride_qq+d_indices[None,:]*stride_qd
    mask_q=(q_indices<N_QUERIES)[:,None]
    Q=tl.load(Q_ptr+q_offsets,mask=mask_q)

    o_indices=q_indices
    o_offsets=batch_idx*stride_ob+o_indices[:,None]*stride_oq+d_indices[None,:]*stride_od
    O=tl.load(O_ptr+o_offsets,mask=mask_q)
    dO=tl.load(dO_ptr+o_offsets,mask=mask_q)
    D=tl.sum(O*dO,axis=1)

    L_indices=q_indices
    L_offsets=batch_idx*stride_lb+L_indices*stride_lq
    mask_L=L_indices<N_QUERIES
    L=tl.load(L_ptr+L_offsets,mask=mask_L)

    dQ=tl.zeros((Q_TILE_SIZE,dim),dtype=tl.float32)

    for k_tile_idx in range(0,tl.cdiv(N_KEYS,K_TILE_SIZE)):
        k_indices=k_tile_idx*K_TILE_SIZE+tl.arange(0,K_TILE_SIZE)
        k_offsets=batch_idx*stride_kb+k_indices[:,None]*stride_kk+d_indices[None,:]*stride_kd
        mask_k=(k_indices<N_KEYS)[:,None]
        K=tl.load(K_ptr+k_offsets,mask=mask_k)

        v_indices=k_indices
        v_offsets=batch_idx*stride_vb+v_indices[:,None]*stride_vk+d_indices[None,:]*stride_vd
        V=tl.load(V_ptr+v_offsets,mask=mask_k)

        S=tl.dot(Q,tl.trans(K))*scale
        if IS_CAUSAL:
            mask=q_indices[:,None]>=k_indices[None,:]
            S=tl.where(mask,S,float("-inf"))

        P=tl.exp(S-L[:,None])
        dP=tl.dot(dO,tl.trans(V))
        dS=P*(dP-D[:,None])
        dQ=dQ+tl.dot(dS,K)*scale

    tl.store(dQ_ptr+q_offsets,dQ,mask=mask_q)

@triton.jit
def attn_bwd_kernel2(
    Q_ptr,K_ptr,V_ptr,O_ptr,L_ptr,
    dO_ptr,dK_ptr,dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    dim: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL:tl.constexpr,
):
    k_tile_idx=tl.program_id(0)
    batch_idx=tl.program_id(1)

    k_indices=k_tile_idx*K_TILE_SIZE+tl.arange(0,K_TILE_SIZE)
    d_indices=tl.arange(0,dim)
    k_offsets=batch_idx*stride_kb+k_indices[:,None]*stride_kk+d_indices[None,:]*stride_kd
    mask_k=(k_indices<N_KEYS)[:,None]
    K=tl.load(K_ptr+k_offsets,mask=mask_k)

    v_indices=k_indices
    v_offsets=batch_idx*stride_vb+v_indices[:,None]*stride_vk+d_indices[None,:]*stride_vd
    V=tl.load(V_ptr+v_offsets,mask=mask_k)

    dK=tl.zeros((K_TILE_SIZE,dim),dtype=tl.float32)
    dV=tl.zeros((K_TILE_SIZE,dim),dtype=tl.float32)

    for q_tile_idx in range(0,tl.cdiv(N_QUERIES,Q_TILE_SIZE)):
        q_indices=q_tile_idx*Q_TILE_SIZE+tl.arange(0,Q_TILE_SIZE)
        q_offsets=batch_idx*stride_qb+q_indices[:,None]*stride_qq+d_indices[None,:]*stride_qd
        mask_q=(q_indices<N_QUERIES)[:,None]
        Q=tl.load(Q_ptr+q_offsets,mask=mask_q)

        o_indices=q_indices
        o_offsets=batch_idx*stride_ob+o_indices[:,None]*stride_oq+d_indices[None,:]*stride_od
        O=tl.load(O_ptr+o_offsets,mask=mask_q)
        dO=tl.load(dO_ptr+o_offsets,mask=mask_q)
        D=tl.sum(O*dO,axis=1)

        L_indices=q_indices
        L_offsets=batch_idx*stride_lb+L_indices*stride_lq
        mask_L=L_indices<N_QUERIES
        L=tl.load(L_ptr+L_offsets,mask=mask_L)

        S=tl.dot(Q,tl.trans(K))*scale
        if IS_CAUSAL:
            mask=q_indices[:,None]>=k_indices[None,:]
            S=tl.where(mask,S,float("-inf"))
        P=tl.exp(S-L[:,None])

        dP=tl.dot(dO,tl.trans(V))
        dS=P*(dP-D[:,None])
        dV+=tl.dot(tl.trans(P),dO)
        dK+=tl.dot(tl.trans(dS),Q)*scale

    tl.store(dK_ptr+k_offsets,dK,mask=mask_k)
    tl.store(dV_ptr+v_offsets,dV,mask=mask_k)

    
class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,is_causal=False):
        batch_size,n_queries,D=q.shape
        n_keys=k.shape[1]
        scale=1/D**0.5

        q_tile_size=32
        k_tile_size=32

        o=torch.empty_like(q)
        L=torch.empty((batch_size,n_queries),device=q.device,dtype=torch.float32)

        grid=(triton.cdiv(n_queries,q_tile_size),batch_size)

        attn_fwd_kernel[grid](
            q,k,v,o,L,
            q.stride(0),q.stride(1),q.stride(2),
            k.stride(0),k.stride(1),k.stride(2),
            v.stride(0),v.stride(1),v.stride(2),
            o.stride(0),o.stride(1),o.stride(2),
            L.stride(0),L.stride(1),
            n_queries,n_keys,
            scale,
            D=D,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            IS_CAUSAL=is_causal,
        )

        ctx.save_for_backward(q,k,v,o,L)
        ctx.is_causal=is_causal

        return o

    @staticmethod
    def backward(ctx,do):
        q,k,v,o,L=ctx.saved_tensors
        is_causal=ctx.is_causal

        dQ=torch.zeros_like(q)
        dK=torch.zeros_like(k)
        dV=torch.zeros_like(v)

        batch_size,n_queries,dim=q.shape
        n_keys=k.shape[1]

        q_tile_size=32
        k_tile_size=32
        scale=1/dim**0.5

        grid1=(triton.cdiv(n_queries,q_tile_size),batch_size)

        attn_bwd_kernel1[grid1](
            q,k,v,o,L,
            do,dQ,
            q.stride(0),q.stride(1),q.stride(2),
            k.stride(0),k.stride(1),k.stride(2),
            v.stride(0),v.stride(1),v.stride(2),
            o.stride(0),o.stride(1),o.stride(2),
            L.stride(0),L.stride(1),
            n_queries,n_keys,
            scale,
            dim=dim,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            IS_CAUSAL=is_causal,
        )

        grid2=(triton.cdiv(n_keys,k_tile_size),batch_size)

        attn_bwd_kernel2[grid2](
            q,k,v,o,L,
            do,dK,dV,
            q.stride(0),q.stride(1),q.stride(2),
            k.stride(0),k.stride(1),k.stride(2),
            v.stride(0),v.stride(1),v.stride(2),
            o.stride(0),o.stride(1),o.stride(2),
            L.stride(0),L.stride(1),
            n_queries,n_keys,
            scale,
            dim=dim,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            IS_CAUSAL=is_causal,
        )

        return dQ,dK,dV,None

