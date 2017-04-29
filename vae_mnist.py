from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from model import VariationalAutoEncoder, VAELoss


class Session():
    def __init__(self, model, lr=1e-3, is_cuda=False):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = VAELoss()

    def train(self, loader, epoch_number):
        # built-in method for the nn.module, sets a training flag.
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(loader):
            data = Variable(data)
            # do not use CUDA atm
            self.optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(data)
            loss = self.loss_fn(data, mu, log_var, recon_batch)
            loss.backward()
            train_loss += loss.data[0]
            self.optimizer.step()

    def test(self, loader):
        # nn.Module method, sets the training flag to False
        self.model.eval()
        test_loss = 0
        for batch_idx, (data, _) in enumerate(loader):
            data = Variable(data, volatile=True)
            # do not use CUDA atm
            recon_batch, mu, logvar = self.model(data)
            test_loss += self.loss_fn(data, mu, logvar, recon_batch).data[0]

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


EPOCHS = 2
BATCH_SIZE = 128
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **{})
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **{})

for epoch in range(1, EPOCHS + 1):
    vae = VariationalAutoEncoder()
    sess = Session(vae, lr=13e-3)
    sess.train(train_loader, epoch)
    print('epoch {} complete'.format(epoch))
    sess.test(test_loader)
