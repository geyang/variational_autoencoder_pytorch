import argparse
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

from model import VariationalAutoEncoder
from vae_mnist import Session

parser = argparse.ArgumentParser(description='variational autoencoder training')

parser.add_argument('--prefix', type=str, default='VAE', help='the prefix of this session')
parser.add_argument('--dashboard-server', default=None, help='batch size for test')
parser.add_argument('--dataset', '-d', type=lambda s: s.upper(), default='mnist',
                    help='torchvision dataset key: choose from [mnist, ...]')
parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/{prefix}-{date}-{time}.pkl',
                    help='path for the checkpoint file')
parser.add_argument('-L', '--load', type=bool, default=False, help='load model at the beginning')
parser.add_argument('-S', '--save', type=bool, default=False, help='save model on exit')
parser.add_argument('--batch-size', type=int, default=128, help='batch size for training and test')
parser.add_argument('--test-batch-size', type=int, default=None, help='batch size for test')
parser.add_argument('--epochs', type=int, default=20, help='batch size for test')
parser.add_argument('--start', type=int, default=0, help='batch size for test')
parser.add_argument('--verbose', type=bool, default=True, help='print batch loss info')
parser.add_argument('--report-interval', type=int, default=100, help='report interview in verbose mode')


# model parameters
parser.add_argument('--latent-n', type=int, default=20, help='latent size of the encoder')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate during training')

args = parser.parse_args()
print("latent_n is: {}".format(args.latent_n))

train_loader = torch.utils.data.DataLoader(
    getattr(datasets, args.dataset)('../data', train=True, download=True,
                                    transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **{})

test_loader = torch.utils.data.DataLoader(
    getattr(datasets, args.dataset)('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **{})

vae = VariationalAutoEncoder(latent_n=args.latent_n)

with Session(vae, prefix=args.prefix, lr=args.learning_rate, dashboard_server=args.dashboard_server, checkpoint_path=args.checkpoint_path,
             load=args.load, save=args.save) as sess:
    for epoch in range(args.epochs):
        sess.train(train_loader, epoch, verbose=args.verbose, report_interval=args.report_interval)
        print('epoch {} complete'.format(epoch))
        sess.test(test_loader)
