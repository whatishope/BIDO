import torch
import torch.nn.functional as F
from torch import nn

class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim=model_dim
        self.num_heads=num_heads
        self.mask=mask

        self.head_dim=model_dim//num_heads

        self.FC_Q=nn.Linear(model_dim, self.head_dim*num_heads)
        self.FC_K=nn.Linear(model_dim, self.head_dim*num_heads)
        self.FC_V=nn.Linear(model_dim, self.head_dim*num_heads)

        self.out_proj=nn.Linear(self.head_dim*num_heads, model_dim)

    def forward(self, query, key, value):
        # Q, k, v: [batch_size, num_attention, model_dim]
        batch_size=query.shape[0]

        query=self.FC_Q(query)
        key=self.FC_K(key)
        value=self.FC_V(value)

        query=torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key=torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value=torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key=key.transpose(-1,-2)

        attn_score=(query@key)/self.head_dim**0.5

        attn_score=torch.softmax(attn_score, dim=-1)
        out=attn_score@value
        out=torch.cat(torch.split(out, batch_size, dim=0), dim=-1)

        out=self.out_proj(out)

        return out
    
class SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False):
        super().__init__()

        self.attn=AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward=nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1=nn.LayerNorm(model_dim)
        self.ln2=nn.LayerNorm(model_dim)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)

    def forward(self, x):
        # input shape [batch_size, num_attention, model_dim]
        residual=x 
        out=self.attn(x,x,x)
        out=self.dropout1(out)
        out=self.ln1(residual+out)

        residual=out
        out=self.feed_forward(out)
        out=self.dropout2(out)
        out=self.ln2(residual+out)

        return out
    

if __name__ == '__main__':
    x = torch.randn([12, 33, 768])
    rca = SelfAttentionLayer(model_dim=768, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False)
    out,  attn_score = rca(x)
    print(out.shape)  # 12, 33, 768
    print(attn_score.shape)
