import argparse
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

from model import VariationalAutoEncoder
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from vae_mnist import Session

parser = argparse.ArgumentParser(description='variational autoencoder training')

parser.add_argument('--prefix', type=str, default='VAE', help='the prefix of this session')
parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/{prefix}-{date}-{time}.pkl',
                    help='path for the checkpoint file')
parser.add_argument('--row', type=int, default=10, help='columns in the output')
parser.add_argument('--col', type=int, default=10, help='rows in the output')
parser.add_argument('--output', type=str, default="./figures/{prefix}-{date}-{time}.png")

# model parameters
parser.add_argument('--latent-n', type=int, default=20, help='latent size of the encoder')

args = parser.parse_args()

vae = VariationalAutoEncoder(latent_n=args.latent_n)
with Session(vae, prefix=args.prefix, checkpoint_path=args.checkpoint_path, load=True) as sess:
    row, col = args.row, args.col
    z = Variable(torch.randn(row * col, args.latent_n))
    x = vae.decoder(z)

    fig = plt.figure(figsize=(15, 15))
    for n in range(row * col):
        plt.subplot(row, col, n + 1)
        plt.imshow(x[n].view(28, 28).data.numpy(), cmap='gray', aspect='auto')

    plt.savefig(args.output.format(**sess.format_dict), dpi=300, bbox_inches="tight")
