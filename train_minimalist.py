# from statistics import mean
import torch
import torchvision
# from torchvision import transforms
import argparse
from model import Generator

batch_size = 8

generator_args = argparse.Namespace(
    latent_dim = 512,
    n_mlp = 8,
    g_size = 256,
    channel_multiplier = 2,
    start_iter = 0,
    ckpt = "./550000.pt"
)

print("load model:", generator_args.ckpt)

ckpt = torch.load(generator_args.ckpt, map_location=lambda storage, loc: storage)

device = "cuda"
generator = Generator(
    generator_args.g_size, generator_args.latent_dim, generator_args.n_mlp, channel_multiplier=generator_args.channel_multiplier
).to(device)
generator.load_state_dict(ckpt["g"], strict=False)

noise = torch.randn(batch_size, generator_args.latent_dim, device=device)
fake_img, _ = generator([noise])
print(fake_img.shape)

# normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# print(normalize(fake_img))
torchvision.utils.save_image(fake_img, "./test.png", normalize=True, range=(-1,1))