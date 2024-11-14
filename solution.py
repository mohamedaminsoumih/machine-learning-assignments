import random
import numpy as np
import torch
from typing import Tuple, List, NamedTuple
from tqdm import tqdm
import torchvision
from torchvision import transforms

# Seed all random number generators
np.random.seed(197331)
torch.manual_seed(197331)
random.seed(197331)

class NetworkConfiguration(NamedTuple):
    n_channels: Tuple[int, ...] = (16, 32, 48) 
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (1, 1, 1)
    dense_hiddens: Tuple[int, ...] = (256, 256)

class Trainer:
    def __init__(self,
                 network_type: str = "mlp",
                 net_config: NetworkConfiguration = NetworkConfiguration(),
                 lr: float = 0.001,
                 batch_size: int = 128,
                 activation_name: str = "relu"):

        self.lr = lr
        self.batch_size = batch_size
        self.train, self.test = self.load_dataset(self)
        dataiter = iter(self.train)
        images, labels = next(dataiter)
        input_dim = images.shape[1:]
        self.network_type = network_type
        activation_function = self.create_activation_function(activation_name)
        if network_type == "mlp":
            self.network = self.create_mlp(input_dim[0]*input_dim[1]*input_dim[2], 
                                           net_config,
                                           activation_function)
        elif network_type == "cnn":
            self.network = self.create_cnn(input_dim[0], 
                                           net_config, 
                                           activation_function)
        else:
            raise ValueError("Network type not supported")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.train_logs = {'train_loss': [], 'test_loss': [],
                           'train_mae': [], 'test_mae': []}

    @staticmethod
    def load_dataset(self):
        transform = transforms.ToTensor()

        trainset = torchvision.datasets.SVHN(root='./data', split='train',
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                shuffle=True)

        testset = torchvision.datasets.SVHN(root='./data', split='test',
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                shuffle=False)

        return trainloader, testloader

    @staticmethod
    def create_mlp(input_dim: int, net_config: NetworkConfiguration,
                   activation: torch.nn.Module) -> torch.nn.Module:
        layers = []
        current_dim = input_dim
        for hidden_dim in net_config.dense_hiddens:
            layers.append(torch.nn.Linear(current_dim, hidden_dim))
            layers.append(activation)
            current_dim = hidden_dim
        layers.append(torch.nn.Linear(current_dim, 10))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def create_cnn(in_channels: int, net_config: NetworkConfiguration,
                   activation: torch.nn.Module) -> torch.nn.Module:
        layers = []
        current_in_channels = in_channels
        for out_channels, kernel_size, stride in zip(net_config.n_channels, 
                                                      net_config.kernel_sizes, 
                                                      net_config.strides):
            layers.append(torch.nn.Conv2d(current_in_channels, out_channels, 
                                          kernel_size=kernel_size, stride=stride, padding=1))
            layers.append(activation)
            layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            current_in_channels = out_channels
        
        layers.append(torch.nn.Flatten())
        conv_output_size = current_in_channels * (32 // (2 ** len(net_config.n_channels))) ** 2
        layers.append(torch.nn.Linear(conv_output_size, net_config.dense_hiddens[0]))
        layers.append(activation)
        layers.append(torch.nn.Linear(net_config.dense_hiddens[0], 10))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def create_activation_function(activation_str: str) -> torch.nn.Module:
        if activation_str.lower() == "relu":
            return torch.nn.ReLU()
        elif activation_str.lower() == "tanh":
            return torch.nn.Tanh()
        elif activation_str.lower() == "sigmoid":
            return torch.nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation_str}")

    def compute_loss_and_mae(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.network(X)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, y.long())
        with torch.no_grad():
            _, predictions = torch.max(outputs, 1)
            mae = torch.mean(torch.abs(predictions - y))
        return loss, mae

    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        self.network.train()
        self.optimizer.zero_grad()
        loss, mae = self.compute_loss_and_mae(X_batch, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item(), mae.item()

    def train_loop(self, n_epochs: int) -> dict:
        N = len(self.train)
        for epoch in tqdm(range(n_epochs)):
            train_loss = 0.0
            train_mae = 0.0
            for i, data in enumerate(self.train):
                inputs, labels = data
                loss, mae = self.training_step(inputs, labels)
                train_loss += loss
                train_mae += mae
            self.train_logs['train_mae'].append(train_mae / N)
            self.train_logs['train_loss'].append(train_loss / N)
            self.evaluation_loop()
        return self.train_logs

    def evaluation_loop(self) -> None:
        self.network.eval()
        N = len(self.test)
        with torch.inference_mode():
            test_loss = 0.0
            test_mae = 0.0
            for data in self.test:
                inputs, labels = data
                loss, mae = self.compute_loss_and_mae(inputs, labels)
                test_loss += loss.item()
                test_mae += mae.item()
        self.train_logs['test_mae'].append(test_mae / N)
        self.train_logs['test_loss'].append(test_loss / N)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.network.eval()
        with torch.inference_mode():
            loss, mae = self.compute_loss_and_mae(X, y)
        return loss.item(), mae.item()
