from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Categorical

class SurrogateAddressTransition(nn.Module):

    def __init__(self, input_shape, next_address=None, num_layers=2,
                 first_address=False, last_address=False, hidden_size=None):
        super().__init__()
        self._n_classes = 1
        self._input_shape = util.to_size(input_shape)
        self._output_shape = torch.Size([self._n_classes + 1])

        self._input_dim = util.prod(self._input_shape)
        self._output_dim = util.prod(self._output_shape)

        self._first_address = first_address
        self._last_address = last_address
        if next_address:
            self._addresses = [next_address]
            self._address_to_class = {next_address: torch.Tensor([0])}
        else:
            self._addresses = ["__end"]
            self._address_to_class = {"__end": torch.Tensor([0])}
        self._ff = nn.ModuleDict()
        if num_layers > 1:
            if hidden_size:
                hidden_size = hidden_size
            else:
                hidden_size = int((self._input_dim + self._output_dim)/2)
            self._class_layer_input = hidden_size
            layers = [("layer0", nn.Linear(self._input_dim, hidden_size)),
                      ("relu0", nn.ReLU())]
            for n in range(num_layers - 2):
                layers.append((f"layer{n+1}", nn.Linear(hidden_size, hidden_size)))
                layers.append((f"relu{n+1}", nn.ReLU()))
            self._ff["embedding"] = nn.Sequential(OrderedDict(layers))
        else:
            self._class_layer_input = self._input_dim

        self._ff["class_layer"] = nn.Linear(self._class_layer_input, self._output_dim)

        self._softmax = nn.Softmax(dim=1)

        self._total_train_iterations = 0

    def forward(self, x):
        if self._first_address:
            self._categorical = AddressCategorical(probs=[1], n_classes=self._n_classes, transform=self._transform_to_address)
        elif self._last_address:
            self._categorical = Categorical(probs=[1])
            self._categorical = AddressCategorical(probs=[1], n_classes=self._n_classes, transform=self._transform_to_address)
        else:
            batch_size = x.size(0)
            x = self._ff["embedding"](x)
            x = self._ff["class_layer"](x)
            self.probs = self._softmax(x)
            self._categorical = AddressCategorical(probs=self.probs,
                                                   n_classes=self._n_classes,
                                                   transform=self._transform_to_address)

        return self._categorical

    def add_address_transition(self, new_address):
        classes_param= {"cw": self._ff["class_layer"].weight.data,
                        "cb": self._ff["class_layer"].bias.data}
        placeholder_param = {"pw": self._ff["class_layer"].weight.data[self._n_classes:,:],
                             "pb": self._ff["class_layer"].bias.data[self._n_classes:]}

        new_weights = torch.cat([classes_param["cw"], placeholder_param["pw"]], dim=0)
        new_bias = torch.cat([classes_param["cb"], placeholder_param["pb"]], dim=0)

        # since we zero index the number we use to encode the address, we can use the previous n_classes
        # before we update n_classes
        self._address_to_class[new_address] = torch.Tensor([self._n_classes])
        self._addresses.append(new_address)
        self._n_classes += 1
        self._output_shape = torch.Size([self._n_classes + 1])
        def init_new_params(m):
            # replace the weights and biasses (TODO: ADD A TINY BIT OF NOISE??)
            if type(m) == nn.Linear:
                m.weight = nn.Parameter(new_weights)
                m.bias = nn.Parameter(new_bias)

        self._ff["class_layer"] = nn.Linear(self._class_layer_input, self._output_dim).apply(init_new_params).to(device=util._device)

    def loss(self, next_addresses):
        classes = self._transform_to_class(next_addresses).to(device=util._device)
        loss = -self._categorical.log_prob(classes)
        return loss

    def _transform_to_class(self, next_addresses):
        return torch.Tensor([self._address_to_class[next_address] for next_address in next_addresses])

    def _transform_to_address(self, address_class):
        return self._addresses[address_class]

class AddressCategorical(Categorical):
    def __init__(self, probs=None, n_classes=0, transform=None):
        super().__init__(probs=probs)
        self._transform_to_address = transform
        self._n_classes = n_classes

    def sample(self):
        c = super().sample()
        if c == self._n_classes:
            return "__unknown"
        else:
            return self._transform_to_address(c)
