import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

class FeedForward(nn.Module):
    """Customizable fully-connected feedforward network
    for acoustic model learning.
    """
    def __init__(
        self,
        hidden_size=2,
        n_layers=1,
        input_size=1,
        output_size=1,
        p=0.0,
        activation='ReLU',
        bias=True,
        ):

        super(FeedForward, self).__init__()

        self.n_layers = n_layers

        size  = [input_size] + [hidden_size,] * (self.n_layers-1) + [output_size]
        self.layers = [nn.Linear(size[i], size[i+1], bias=bias) for i in range(self.n_layers)]

        # stack all layers
        self.layers = nn.ModuleList(self.layers)

        # dropout layer
        self.dropout = nn.Dropout(p=p)

        # activation
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise Exception('{} not an available activation'.format(activation))

    def forward(self, x):
        if x.is_contiguous():
            x = x.view(x.shape[0], -1)
        else:
            x = x.reshape(x.shape[0], -1)
            
        for i in range(self.n_layers-1):
            x = self.dropout(self.activation(self.layers[i](x)))
        x = nn.Sigmoid()(self.layers[-1](x))
        return x


class RNN(nn.Module):
    """RNN with probability output to learn sequence modelling.
    """
    def __init__(self, input_size, hidden_size, n_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_layers = n_layers

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.n_layers)
        self.head = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x):
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = nn.Sigmoid()(self.head(out))

        return out


class TrainMgr(nn.Module):
    """Manage model training.
    """
    def __init__(self,
                 config_acoustic_model={},
                 config_language_model={},
                 use_cuda=False,
                 lr=0.01,
                 save_every=None,
                 tol=1e-8,
                ):
        super(TrainMgr, self).__init__()

        self.use_cuda = use_cuda

        # Config dict for underlying models
        self.config_acoustic_model = config_acoustic_model
        self.config_language_model = config_language_model

        if len(self.config_acoustic_model):
            self.acoustic_model = FeedForward(
                input_size=self.config_acoustic_model['input_size'],
                hidden_size=self.config_acoustic_model['hidden_size'],
                n_layers=self.config_acoustic_model['n_layers'],
                output_size=self.config_acoustic_model['output_size'],
                activation=self.config_acoustic_model.get('activation', 'ReLU'),
                p=self.config_acoustic_model.get('p', 0.0),
            ).to(self.device)
            self.lr_acoustic = self.config_acoustic_model.get('lr', 0.01)
            self.batch_size_acoustic = self.config_acoustic_model.get('batch_size', 100)
            self.optimizer_acoustic_model = torch.optim.Adam(self.acoustic_model.parameters(), lr=self.lr_acoustic)

        if len(self.config_language_model):
            self.language_model = RNN(
                input_size=self.config_language_model['input_size'],
                hidden_size=self.config_language_model['hidden_size'],
                n_layers=self.config_language_model['n_layers'],
            ).to(self.device)
            self.lr_language = self.config_language_model.get('lr', 0.01)
            self.batch_size_language = self.config_language_model.get('batch_size', 100)
            self.optimizer_language_model = torch.optim.Adam(self.language_model.parameters(), lr=self.lr_language)

        # If not None. save neural network to .pt every few epochs
        self.save_every = save_every

        # Numerical tolerance (e.g for proba to be between 0.0 and 1.0)
        self.tol = tol

    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

    def train_acoustic(self, data, n_epochs):
        """
        Arguments
        ---------
        data : TensorDataset
        """
        loader = DataLoader(data, batch_size=self.batch_size_acoustic, shuffle=True)

        tdqm_dict_keys = ['loss']
        tdqm_dict = dict(zip(tdqm_dict_keys, [0.0]))

        self.acoustic_model.train()

        for epoch in range(n_epochs):
            # initialize cumulative losses to zero at the start of epoch
            total_loss = 0.0
            with tqdm(total=len(loader),
                      unit_scale=True,
                      postfix={'loss': 0.0},
                      desc="Epoch : %i/%i" % (epoch+1, n_epochs),
                      ncols=100
                     ) as pbar:
                for batch_idx, (batch, target) in enumerate(loader):
                    batch = batch.to(self.device)

                    y = self.acoustic_model(batch)

                    y = torch.clamp(y, self.tol, 1.0 - self.tol)

                    loss = F.binary_cross_entropy(y, target)

                    self.optimizer_acoustic_model.zero_grad()
                    loss.backward()
                    self.optimizer_acoustic_model.step()

                    total_loss += loss.item()

                    # Logging
                    tdqm_dict['loss'] = total_loss / (batch_idx+1)
                    pbar.set_postfix(tdqm_dict)
                    pbar.update(1)

                    if self.save_every and (epoch+1) % self.save_every == 0:
                        torch.save(self.acoustic_model.state_dict(), 'acoustic_model_{}.pt'.format(epoch+1))

    def train_language(self, data_list, n_epochs):
        """
        Arguments
        ---------
        data_list : list
        """
        idx_seq = list(range(len(data_list)))

        tdqm_dict_keys = ['loss']
        tdqm_dict = dict(zip(tdqm_dict_keys, [0.0]))

        self.language_model.train()

        for epoch in range(n_epochs):
            np.random.shuffle(idx_seq)
            loader = np.array_split(idx_seq, len(data_list) // self.batch_size_language)
            # initialize cumulative losses to zero at the start of epoch
            total_loss = 0.0
            with tqdm(total=len(loader),
                      unit_scale=True,
                      postfix={'loss': 0.0},
                      desc="Epoch : %i/%i" % (epoch+1, n_epochs),
                      ncols=100
                     ) as pbar:
                for batch_idx, batch in enumerate(loader):
                    batch = [data_list[i] for i in batch]

                    loss = torch.tensor([0.0])
                    for i, y in enumerate(batch):
                        y_pred = self.language_model(y.unsqueeze(0))
                        y_pred = torch.clamp(y_pred, self.tol, 1.0 - self.tol).squeeze()
                        loss += F.binary_cross_entropy(y_pred, y)

                    self.optimizer_language_model.zero_grad()
                    loss.backward()
                    self.optimizer_language_model.step()

                    total_loss += loss.item()

                    # Logging
                    tdqm_dict['loss'] = total_loss / (batch_idx+1)
                    pbar.set_postfix(tdqm_dict)
                    pbar.update(1)

                    if self.save_every and (epoch+1) % self.save_every == 0:
                        torch.save(self.language_model.state_dict(), 'language_model_{}.pt'.format(epoch+1))
