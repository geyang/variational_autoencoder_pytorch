# Variational Autoencoder (implementation in pyTorch) [![](https://img.shields.io/badge/link_on-GitHub-brightgreen.svg?style=flat-square)](https://github.com/episodeyang/variational_autoencoder_pytorch)

This is implemented using the pyTorch tutorial example as a reference.

### Todo
- [ ] add `pip` build chain
- [ ] read [Graphical Model]()

## Usage (To Run)
1. install dependencies via
    ```bash
    pip install -r requirement.txt
    ```
2. Fire up a `visdom` server instance to show the visualizations. Run in a dedicated prompt to keep this alive.
    ```bash
    python -m visdom.server
    ```
3. In a new prompt run
    ```bash
    python vae_mnist.py
    ```
    
## Variational Autoencoder (VAE)

Going through the code is almost the best way to explain the Variational Autoencoder. VAE has four parts:
1. Encoder
2. Decoder
3. Reparametrization in-between
4. The variational loss function

Here we will talk about them one by one. For details of the model and motivations, you can take a look at the 
original paper: https://arxiv.org/abs/1312.6114

### Encoder

The code looks like this:

```python
class Encoder(nn.Module):
    def __init__(self, hidden_n_1=400, hidden_n_2=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, hidden_n_1)
        self.fc_mu = nn.Linear(hidden_n_1, hidden_n_2)
        self.fc_var = nn.Linear(hidden_n_1, hidden_n_2)

    def forward(self, x):
        h1 = self.fc1(x)
        return self.fc_mu(h1), self.fc_var(h1)
```
Notice here that we don't have any non-linear component.

### Decoder

```python
class Decoder(nn.Module):
    def __init__(self, hidden_n_1=20, hidden_n_2=400):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_n_1, hidden_n_2)
        self.fc2 = nn.Linear(hidden_n_2, 784)

        self.reLU = nn.ReLU()  # reLU non-linear unit for the hidden output
        self.sigmoid = nn.Sigmoid()  # sigmoid non-linear unit for the output

    def forward(self, embedded):
        h1 = self.reLU(self.fc1(embedded))
        return self.sigmoid(self.fc2(h1))
```

Note that the decoder has non-linearity for both `h1` and `output` layer. The sigmoid forces the output to be between
0 and 1. Otherwise the cross-entropy loss function gives a `nan` value.

### Reparametrization

Here you want to take the *mean* and *variance* from the encoder, and randomly 
generate an embedded vector according to those parameters. 

In pyTorch:

```python
    def reparametrize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        vector_size = log_var.size()
        eps = Variable(torch.FloatTensor(vector_size).normal_())
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)
```

You can now feed this into the decoder.

The full code of the Variational Autoencoder blow:

```python
class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.reparametrize(mu, log_var)
        return self.decoder(z), mu, log_var

    def reparametrize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        vector_size = log_var.size()
        eps = Variable(torch.FloatTensor(vector_size).normal_())
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)
```

Now, 