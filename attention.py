import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math

def attention(q, k, v, 
              embed_dim, num_heads, 
              in_proj_weight, in_proj_bias,
              out_proj_weight, out_proj_bias,
              batch_first=True):
    
    # transpose if batch first
    if batch_first:
        q = q.transpose(1,0)
        k = k.transpose(1,0)
        v = v.transpose(1,0)
        
    # get dimensions 
    tgt_len, bsz, embed_dim = q.shape
    src_len, _, _ = k.shape
    head_dim = embed_dim // num_heads
    
    # chunk in projection weights
    w_q, w_k, w_v = in_proj_weight.chunk(3)
    b_q, b_k, b_v = in_proj_bias.chunk(3)
    
    # compute in projections
    q = F.linear(q, w_q, b_q) 
    k = F.linear(k, w_k, b_k)
    v = F.linear(v, w_v, b_v)
    
    # reshape for attention 
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    
    # get updated dimensions 
    src_len = k.size(1)
    B, Nt, E = q.shape

    # scale query
    q_scaled = q * math.sqrt(1.0 / float(E))
    
    # compute attention weights
    attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    
    # compute attention output
    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    # average attention weights between heads
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    attn_output_weights = attn_output_weights.mean(dim=1)
    
    # if batch first, reshape output
    if batch_first:
        attn_output = attn_output.transpose(1,0)
    
    return attn_output, attn_output_weights

embed_dim = 8
num_heads = 1
batch_size = 2
seq_len = 5

Q = torch.randn(batch_size, seq_len, embed_dim)
K = torch.randn(batch_size, seq_len, embed_dim)
V = torch.randn(batch_size, seq_len, embed_dim)

multihead_attn = MultiheadAttention(embed_dim=embed_dim, 
                                    num_heads=num_heads, 
                                    batch_first=True)

attn_output_pytorch, attn_output_weights_pytorch = multihead_attn(Q, K, V)

attn_output_custom, attn_output_weights_custom = attention(Q, K, V, 
                                                           embed_dim, 
                                                           num_heads, 
                                                           multihead_attn.in_proj_weight, 
                                                           multihead_attn.in_proj_bias,
                                                           multihead_attn.out_proj.weight, 
                                                           multihead_attn.out_proj.bias,
                                                           batch_first=True)

assert torch.allclose(attn_output_custom, attn_output_pytorch, rtol=1e-6, atol=1e-8), "Attention output does not match."
assert torch.allclose(attn_output_weights_custom, attn_output_weights_pytorch, rtol=1e-6, atol=1e-8), "Attention weights do not match."