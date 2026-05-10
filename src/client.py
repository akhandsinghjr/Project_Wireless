import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict

# Global state to persist local heads across rounds
local_heads_registry = {}

class FedRepClient(fl.client.NumPyClient):
    def __init__(self, cid, base_model, head_model, trainloader, valloader, config, device):
        self.cid = cid
        self.base_model = base_model.to(device)
        self.head_model = head_model.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.config = config
        self.device = device
        
        if self.cid in local_heads_registry:
            self.head_model.load_state_dict(local_heads_registry[self.cid])

    def set_parameters(self, parameters):
        params_dict = zip(self.base_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.base_model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.base_model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        
        lr = self.config['learning_rate']
        momentum = self.config['momentum']
        optimizer_head = torch.optim.SGD(self.head_model.parameters(), lr=lr, momentum=momentum)
        optimizer_base = torch.optim.SGD(self.base_model.parameters(), lr=lr, momentum=momentum)

        # 1. Train Head (Freeze Base)
        self.base_model.requires_grad_(False)
        self.head_model.requires_grad_(True)
        self._train_loop(optimizer_head, criterion, self.config['local_epochs_head'])

        # 2. Train Base (Freeze Head)
        self.base_model.requires_grad_(True)
        self.head_model.requires_grad_(False)
        self._train_loop(optimizer_base, criterion, self.config['local_epochs_base'])

        local_heads_registry[self.cid] = self.head_model.state_dict()
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def _train_loop(self, optimizer, criterion, epochs):
        self.base_model.train()
        self.head_model.train()
        for _ in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.head_model(self.base_model(images))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.base_model.eval()
        self.head_model.eval()
        loss, correct, total = 0.0, 0, 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.head_model(self.base_model(images))
                loss += criterion(outputs, labels).item()
                correct += (outputs.max(1)[1] == labels).sum().item()
                total += labels.size(0)
                
        accuracy = correct / total
        return float(loss/len(self.valloader)), total, {"accuracy": accuracy}