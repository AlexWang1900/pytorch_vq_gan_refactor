import torch
import torch.nn as nn


class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors#1024
        self.latent_dim = args.latent_dim#256
        self.beta = args.beta#0.25

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)#1024,256
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):# z_hat:([8, 256, 8, 8])
        z = z.permute(0, 2, 3, 1).contiguous()#([8, 8, 8, 256])
        z_flattened = z.view(-1, self.latent_dim)#([512, 256]) 64,256

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))#[512, 1024])

        min_encoding_indices = torch.argmin(d, dim=1)# 512 pick one in 1024 for each in 512
        z_q = self.embedding(min_encoding_indices).view(z.shape)#([8, 8, 8, 256])

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        #z_q = z_q.detach()
        z_q = z + (z_q - z).detach()#([8, 8, 8, 256]) copy gradients,foward = zq,backward = z

        z_q = z_q.permute(0, 3, 1, 2)#([8, 256, 8, 8])

        return z_q, min_encoding_indices, loss
