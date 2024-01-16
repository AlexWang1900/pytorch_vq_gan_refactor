import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
from vqgan import VQGAN


class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()

        self.sos_token = args.sos_token #0

        self.vqgan = self.load_vqgan(args)

        transformer_config = {
            "vocab_size": args.num_codebook_vectors,
            "block_size": 512,
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024
        }
        self.transformer = GPT(**transformer_config)

        self.pkeep = args.pkeep

    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        model.load_checkpoint(args.checkpoint_path) #add by wang for test
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices, p1=8, p2=8):#16,16
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image

    def forward(self, x):#([20, 3, 128, 128])
        _, indices = self.encode_to_z(x)#([20, 64])

        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token#([20, 1]) value = 0
        sos_tokens = sos_tokens.long().to("cuda")

        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))#([20, 64]) 0.5
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)#([20, 64])
        new_indices = mask * indices + (1 - mask) * random_indices# mixing

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)#([20, 65]) start of sentence.

        target = indices#([20, 64])

        logits, _ = self.transformer(new_indices[:, :-1])#([20, 64, 1024])

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):#[1,0],([1, 1]),256,100
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)#[1,1]
        for k in range(steps):
            logits, _ = self.transformer(x)#[1,1,1024]
            logits = logits[:, -1, :] / temperature#[1,1024]

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)#[1,1024]

            probs = F.softmax(logits, dim=-1)#([1, 1024])

            ix = torch.multinomial(probs, num_samples=1)#[1,1]

            x = torch.cat((x, ix), dim=1)
        #([1, 257])
        x = x[:, c.shape[1]:]#[1,256]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x):
        log = dict()

        _, indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        start_indices = indices[:, :indices.shape[1] // 2]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1])
        half_sample = self.z_to_image(sample_indices)

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x.mul(0.5).add(0.5)
        log["rec"] = x_rec.mul(0.5).add(0.5)
        log["half_sample"] = half_sample.mul(0.5).add(0.5)
        log["full_sample"] = full_sample.mul(0.5).add(0.5)

        return log, torch.concat((x.mul(0.5).add(0.5), x_rec.mul(0.5).add(0.5), half_sample.mul(0.5).add(0.5), full_sample.mul(0.5).add(0.5)))
















