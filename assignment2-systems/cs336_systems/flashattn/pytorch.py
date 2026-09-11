import torch

class FlashAttentionPytorch(torch.autograd.Function):
    #必须是staticmethed，且第一个参数是ctx，及上下文数据，用于在forward和backward间传递数据
    @staticmethod
    def forward(ctx,q,k,v,is_causal=False):
        batch_size,n_queries,D=q.shape
        n_keys=k.shape[-2]
        scale=1/D**0.5

        batch_q=32
        batch_k=32

        outputs=torch.zeros(batch_size,n_queries,D,device=q.device,dtype=q.dtype)
        #L这一结果将用于backward计算cross_entropy
        L=torch.zeros(batch_size,n_queries,device=q.device,dtype=q.dtype)

        for i in range(0,n_queries,batch_q):
            q_i=q[:,i:i+batch_q,:]
            M=torch.full((batch_size,q_i.shape[1]),float("-inf"),device=q.device,dtype=q.dtype)
            ACC=torch.zeros((batch_size,q_i.shape[1],D),device=q.device,dtype=q.dtype)
            SUM=torch.zeros((batch_size,q_i.shape[1]),device=q.device,dtype=q.dtype)
            for j in range(0,n_keys,batch_k):
                k_j=k[:,j:j+batch_k,:]
                v_j=v[:,j:j+batch_k,:]
                #(batch_size,batch_q,batch_k)
                qkt=q_i@k_j.transpose(-1,-2)*scale
                if is_causal:
                    #需要根据i,j的位置，确定causal_mask
                    q_start,q_end=i,i+q_i.shape[1]
                    k_start,k_end=j,j+k_j.shape[1]
                    #全1矩阵
                    if q_start>=k_end:
                        pass
                    #全0矩阵
                    if q_end<=k_start:
                        continue
                    q_idx=torch.arange(q_start,q_end,1,device=q.device)
                    k_idx=torch.arange(k_start,k_end,1,device=q.device)
                    #mask(q_len,k_len)
                    mask=q_idx[:,None]>=k_idx[None,:]
                    qkt=qkt.masked_fill(~mask,float("-inf"))
                #更新这一query切片的最大值
                m_old=M
                M=torch.maximum(m_old,qkt.amax(dim=-1))
                #alpha(batch_size,batch_q)
                alpha=(m_old-M).exp()
                #scores(batch_size,batch_q,batch_k)
                scores=(qkt-M.unsqueeze(-1)).exp()
                ACC=alpha.unsqueeze(-1)*ACC+scores@v_j
                SUM=alpha*SUM+scores.sum(dim=-1)

            outputs[:,i:i+q_i.shape[1],:]=ACC/SUM.unsqueeze(-1)
            #返回的版本需要加上最大值，这是因为在交叉熵中，执行-(y-x_max)表示减法,在backward环节可以直接使用
            #在backward中可以一步恢复完整的attention scores
            L[:,i:i+q_i.shape[1]]=SUM.log()+M

        ctx.save_for_backward(q,k,v,outputs,L)
        ctx.is_causal=is_causal

        return outputs
    
    @staticmethod
    def backward(ctx,output_grad):
        q,k,v,outputs,L=ctx.saved_tensors
        is_causal=ctx.is_causal

        batch_size,n_queries,d_k=q.shape
        n_keys=k.shape[-2]
        scale=1/d_k**0.5
        #S:(batch_size,n_queries,n_keys)
        S=q@k.transpose(-1,-2)*scale
        if is_causal:
            q_idx=torch.arange(0,n_queries,1,device=q.device)
            k_idx=torch.arange(0,n_keys,1,device=q.device)
            mask=q_idx[:,None]>=k_idx[None,:]
            S=S.masked_fill(~mask,float("-inf"))
        #L:(batch_size,n_queries)
        P=(S-L.unsqueeze(-1)).exp()
        #outputs:(batch_size,n_queries,d_k）
        D=(outputs*output_grad).sum(dim=-1)

        #output_grad:(batch_size,n_queries,dim)
        v_grad=P.transpose(-1,-2)@output_grad
        p_grad=output_grad@v.transpose(-1,-2)
        s_grad=P*(p_grad-D.unsqueeze(-1))
        q_grad=s_grad@k*scale
        k_grad=s_grad.transpose(-1,-2)@q*scale

        #backward的返回参数个数需要和forward的输入对齐
        return q_grad,k_grad,v_grad,None