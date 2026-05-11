import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict

# A wrapper to combine your Base and Head into one monolithic model
class FullModel(nn.Module):
    def __init__(self, base_model, head_model):
        super().__init__()
        self.base = base_model
        self.head = head_model

    def forward(self, x):
        return self.head(self.base(x))

class FedAvgClient(fl.client.NumPyClient):
    def __init__(self, cid, base_model, head_model, trainloader, valloader, config, device):
        self.cid = cid
        self.device = device
        self.model = FullModel(base_model, head_model).to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.config = config

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        # FedAvg sends the ENTIRE model (Base + Head) back to the server
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        
        # Train both parts together with one optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=self.config['learning_rate'], 
                                    momentum=self.config['momentum'])

        self.model.train()
        # 5 epochs total to match the FedRep (3 head + 2 base)
        epochs = self.config['local_epochs_head'] + self.config['local_epochs_base']
        
        for _ in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                correct += (outputs.max(1)[1] == labels).sum().item()
                total += labels.size(0)
                
        accuracy = correct / total
        return float(loss/len(self.valloader)), total, {"accuracy": accuracy}