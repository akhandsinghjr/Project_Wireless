import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def prepare_dataset_dirichlet(num_clients: int, alpha: float):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    valset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    if alpha == 'IID':
        lengths_train = [len(trainset) // num_clients] * num_clients
        lengths_train[0] += len(trainset) - sum(lengths_train)
        train_subsets = torch.utils.data.random_split(trainset, lengths_train)
        
        lengths_val = [len(valset) // num_clients] * num_clients
        lengths_val[0] += len(valset) - sum(lengths_val)
        val_subsets = torch.utils.data.random_split(valset, lengths_val)
        
        client_train_indices = {i: train_subsets[i].indices for i in range(num_clients)}
        client_val_indices = {i: val_subsets[i].indices for i in range(num_clients)}
        return trainset, valset, client_train_indices, client_val_indices

    train_labels = np.array(trainset.targets)
    val_labels = np.array(valset.targets)
    num_classes = 10
    
    min_size = 0
    min_require_size = 10
    
    print(f"Generating Dirichlet (alpha={alpha}) splits...")
    
    while min_size < min_require_size:
        client_train_indices = {i: [] for i in range(num_clients)}
        client_val_indices = {i: [] for i in range(num_clients)}
        
        for c in range(num_classes):
            idx_c_train = np.where(train_labels == c)[0]
            idx_c_val = np.where(val_labels == c)[0]
            
            np.random.shuffle(idx_c_train)
            np.random.shuffle(idx_c_val)
            
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Apportion Train Data
            counts_train = (proportions * len(idx_c_train)).astype(int)
            counts_train[np.argmax(proportions)] += len(idx_c_train) - counts_train.sum()
            
            # Apportion Val Data
            counts_val = (proportions * len(idx_c_val)).astype(int)
            counts_val[np.argmax(proportions)] += len(idx_c_val) - counts_val.sum()
            
            start_train, start_val = 0, 0
            for i in range(num_clients):
                end_train = start_train + counts_train[i]
                client_train_indices[i].extend(idx_c_train[start_train:end_train])
                start_train = end_train
                
                end_val = start_val + counts_val[i]
                client_val_indices[i].extend(idx_c_val[start_val:end_val])
                start_val = end_val
                
        min_size = min([len(indices) for indices in client_train_indices.values()])

    return trainset, valset, client_train_indices, client_val_indices

def load_client_data(client_id: int, global_trainset, global_valset, client_train_indices, client_val_indices, batch_size=32):
    train_idx = client_train_indices[client_id]
    val_idx = client_val_indices[client_id]
    
    client_trainset = Subset(global_trainset, train_idx)
    client_valset = Subset(global_valset, val_idx) # <-- NOW WE USE LOCAL VAL SET
    
    trainloader = DataLoader(client_trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(client_valset, batch_size=64, shuffle=False)
    
    return trainloader, valloader