import torch
from torch.autograd import Variable
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_n_1=20, hidden_n_2=400):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_n_1, hidden_n_2)
        self.fc2 = nn.Linear(hidden_n_2, 784)

    def forward(self, mu):
        # todo: should take variance also into account?
        h1 = self.fc1(mu)
        return self.fc2(h1)


class Encoder(nn.Module):
    def __init__(self, hidden_n_1=400, hidden_n_2=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, hidden_n_1)
        self.fc_mu = nn.Linear(hidden_n_1, hidden_n_2)
        self.fc_var = nn.Linear(hidden_n_1, hidden_n_2)

    def forward(self, x):
        h1 = self.fc1(x)
        return self.fc_mu(h1), self.fc_var(h1)


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.bce_loss.size_average = False

    # question: how is the loss function using the mu and variance?
    def forward(self, x, mu, log_var, recon_x):
        # assert x.size()[1:] == [784,]
        BCE = self.bce_loss(recon_x, x)
        print(recon_x, x, BCE)
        raise KeyError('stop right here')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        return BCE + KLD


class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.reparametrize(mu, log_var)
        return self.decoder(z), mu, log_var

    # question: what does this reparametrization step do?
    def reparametrize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = Variable(torch.FloatTensor(std.size()).normal_())
        return eps.mul(std).add_(mu)
