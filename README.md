# Variational Autoencoder (implementation in pyTorch) [![](https://img.shields.io/badge/link_on-GitHub-brightgreen.svg?style=flat-square)](https://github.com/episodeyang/variational_autoencoder_pytorch)

This is implemented using the pyTorch tutorial example as a reference.

### Todo
- [ ] theory blog post to explain variational bayesian methods.
- [ ] relate the reparametrization trick to Gumbel-softmax reparametrization trick.

#### Done
- [x] closer look at the paper
- [x] think of a demo: how do you visualize the output of a variational 
- [x] doc: add `requirement.txt`
- [x] reading: read [Graphical Model]()
- [x] data (use torchvision dataloader)
- [x] model (reference pytorch tutorial)
- [x] train 
- [x] chore: add Makefile.
- [x] explain reparameterization
- [x] explain variational loss

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

Or for with a quick shortcut, you can just run `make`. You can take a look at
the [`./Makefile`](Makefile) for more details.

### Using as a command line tool

There are two scripts: 
- `train.py` *and*
- `generate.py`

#### `train.py`
```bash
usage: main.py [-h] [--prefix PREFIX] [--dashboard-server DASHBOARD_SERVER]
               [--dataset DATASET] [--checkpoint-path CHECKPOINT_PATH]
               [-L LOAD] [-S SAVE] [--batch-size BATCH_SIZE]
               [--test-batch-size TEST_BATCH_SIZE] [--epochs EPOCHS]
               [--start START] [--latent-n LATENT_N]
               [--learning-rate LEARNING_RATE]

variational autoencoder training

optional arguments:
  -h, --help            show this help message and exit
  --prefix PREFIX       the prefix of this session
  --dashboard-server DASHBOARD_SERVER
                        batch size for test
  --dataset DATASET, -d DATASET
                        torchvision dataset key: choose from [mnist, ...]
  --checkpoint-path CHECKPOINT_PATH
                        path for the checkpoint file
  -L LOAD, --load LOAD  load model at the beginning
  -S SAVE, --save SAVE  save model on exit
  --batch-size BATCH_SIZE
                        batch size for training and test
  --test-batch-size TEST_BATCH_SIZE
                        batch size for test
  --epochs EPOCHS       batch size for test
  --start START         batch size for test
  --latent-n LATENT_N   latent size of the encoder
  --learning-rate LEARNING_RATE
                        learning rate during training
```

#### `generate.py`

```bash
usage: generate.py [-h] [--prefix PREFIX] [--checkpoint-path CHECKPOINT_PATH]
                   [--row ROW] [--col COL] [--output OUTPUT]
                   [--latent-n LATENT_N]

variational autoencoder generate examples

optional arguments:
  -h, --help            show this help message and exit
  --prefix PREFIX       the prefix of this session
  --checkpoint-path CHECKPOINT_PATH
                        path for the checkpoint file
  --row ROW             columns in the output
  --col COL             rows in the output
  --output OUTPUT
  --latent-n LATENT_N   latent size of the encoder
```

### Experimentation

For experimentation, refer to `./experiment-archive/explore_latent_n.sh`
    
    
## Results:

random digits generated with a 20-dimensional embedding:

![latent-20](./figures/VAE-20170529-20.png)

## Variational Autoencoder (VAE) and Variational Bayesian methods

Going through the code is almost the best way to explain the Variational
Autoencoder. However, to fully understand Variational Bayesian methods and why
it is useful, it is best to first take a look at graphical models plus getting 
a good understanding of various Bayesian inference methods. For the former, I 
recommend Daphne Koller's [Probabilistic Graphical Models course](https://www.youtube.com/playlist?list=PL50E6E80E8525B59C)
from stanford. For the latter, you can take a look at wikipedia or Kevin 
Murphy's textbook.

### Theory Requirements
- Graphical Models
- Maximum Likelihood Estimate and Maximum A Posteriori inference
- Kullback-Leibler divergence

After getting familiar with these concepts, the paper would make a lot more 
sense. 
- arxiv link: https://arxiv.org/abs/1312.6114

### Understanding VAE by reading code

VAE has four parts:

1. Encoder
2. Decoder
3. Reparameterization in-between
4. The variational loss function

The encoder and the decoder doesn't require much explaination. I will show the
code quickly and spend more time with the reparameterization step and the 
variational loss function. 


#### Encoder

The code looks like this: It is a very simple two-layer fully-connected
network with no linearities.

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

#### Decoder

The decoder has non-linearity for both `h1` and `output` layer. The 
sigmoid forces the output to be between 0 and 1. Otherwise the 
cross-entropy loss function gives a `nan` value.

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

#### Reparameterization

Stochastic nodes where the output is drawn from bernoulli and categorical distributions are not differentiable. This 
reparametrization trick is born exactly to solve this problem. The following trick takes the random drawing off the 
main branch of the computation graph, and places it inside a normally distributed random variable. Drawing the following
graph on paper will help you understand it better.

1. first you take the *mean* and *variance* from the encoder
2. draw a random vector from a normal distribution
3. use this random vector as the entropy source, to generate a backward differentiable embedded vector that has the
given *mean* and *variance*.

In pyTorch this would be:

```python
    def reparameterize(self, mu, log_var):
        """you generate a random disteribution w.r.t. the mu and log_var from the embedding space."""
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
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def reparameterize(self, mu, log_var):
        """you generate a random disteribution w.r.t. the mu and log_var from the embedding space."""
        vector_size = log_var.size()
        eps = Variable(torch.FloatTensor(vector_size).normal_())
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)
```

#### Variational Loss Function

The loss function has two parts:
```python
def forward(self, x, mu, log_var, recon_x):
    """gives the batch normalized Variational Error."""
    
    batch_size = x.size()[0]
    BCE = self.bce_loss(recon_x, x)

    KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return (BCE + KLD) / batch_size
```

1. **Mean Square Error Loss** between input and output
    
    the mean square error just takes the mean square difference beween the input and the output of the autoencoder.

2. **KL Divergence Loss** in the embedding layer
    
    The KL divergence loss takes the mean and variance of the embedding vector
    generated by the encoder, and calculates the KL-divergence of that random
    disteribution.
    
    In appendix B from the VAE [paper](https://arxiv.org/abs/1312.6114), 
    $$
    KL = - \frac 1 2 \sum{1 + \log{\sigma_i^2} - \mu_i^2 - \sigma_i^2}.
    $$
    this can be implemented as the following in code:
    ```python
    KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    ```
    
## Demo

we can generate MNIST like numbers by sampling from the embedding layer. Take a look at this ipython notebook: 
[Variationa Auto-Encoder Demo.ipynb](./Variational%20Auto-Encoder%20Demo.ipynb)