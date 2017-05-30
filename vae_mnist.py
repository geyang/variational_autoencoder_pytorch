import numpy as np
import time
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import os

from model import VariationalAutoEncoder, VAELoss

from visdom_helper.visdom_helper import Dashboard


class Session():
    def __init__(self, model, is_cuda=False,
                 train_step_init=0, lr=1e-3, prefix="Variational_Autoencoder",
                 dashboard_server=None,
                 checkpoint_path=None, load=False, save=False):
        self.prefix = prefix
        self.train_step = train_step_init
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = VAELoss()

        if dashboard_server:
            protocol, server, port = dashboard_server.split(':')
            self.dashboard = Dashboard(prefix, server=protocol + ':' + server, port=port)
        elif dashboard_server is False: # when False do not create dashboard instance.
            pass
        else:
            self.dashboard = Dashboard(prefix)

        self._checkpoint_path = checkpoint_path
        self.load = load
        self.save = save
        self.epoch_number = 0

        self.format_dict = dict(prefix=self.prefix,
                                date=time.strftime("%Y%m%d"),
                                time=time.strftime("%H%M%S"))

    def train(self, loader, epoch_number=None, verbose=True, report_interval=100):
        if epoch_number is not None:
            self.epoch_number = epoch_number

        # built-in method for the nn.module, sets a training flag.
        self.model.train()
        losses = []
        for batch_idx, (data, _) in enumerate(loader):
            data = Variable(data)
            # do not use CUDA atm
            self.optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(data)
            loss = self.loss_fn(data, mu, log_var, recon_batch)
            loss.backward()

            loss_data = loss.data.numpy()
            self.optimizer.step()
            self.train_step += 1
            self.dashboard.append('training_loss', 'line',
                                  X=np.array([self.train_step]),
                                  Y=loss_data)

            losses.append(loss_data)

            if verbose and batch_idx % report_interval == 0:
                print('batch loss is: {:.4}'.format(torch.np.mean(losses)/data.size()[0]))

        self.epoch_number += 1
        return losses

    def test(self, test_loader):
        # nn.Module method, sets the training flag to False
        self.model.eval()
        test_loss = 0
        for batch_idx, (data, _) in enumerate(test_loader):
            data = Variable(data, volatile=True)
            # do not use CUDA atm
            recon_batch, mu, logvar = self.model(data)
            test_loss += self.loss_fn(data, mu, logvar, recon_batch).data[0]

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    @property
    def checkpoint_path(self):
        if self._checkpoint_path:
            return self._checkpoint_path.format(**self.format_dict)

    def __enter__(self):
        if self.checkpoint_path and self.load:
            try:
                cp = torch.load(self.checkpoint_path)
                self.model.load_state_dict(cp['state_dict'])
                self.epoch_number = cp['epoch_number']
            except Exception as e:
                print('Did not load checkpoint due to exception: {}'.format(e))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        state_dict = self.model.state_dict()
        path = os.path.split(self.checkpoint_path)
        if len(path) > 1:
            try:
                os.makedirs(os.path.join(*path[:-1]))
            except FileExistsError:
                pass
        if self.checkpoint_path and self.save:
            print('saving state_dict to {}'.format(self.checkpoint_path))
            torch.save(dict(state_dict=state_dict, epoch_number=self.epoch_number), self.checkpoint_path)


if __name__ == "__main__":
    EPOCHS = 20
    BATCH_SIZE = 128

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=True, **{})

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=True, **{})

    losses = []
    vae = VariationalAutoEncoder()
    sess = Session(vae, lr=1e-3)
    for epoch in range(1, EPOCHS + 1):
        losses += sess.train(train_loader, epoch)
        print('epoch {} complete'.format(epoch))
        sess.test(test_loader)
