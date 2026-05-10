import flwr as fl
import yaml
import torch
from src.data import prepare_dataset_dirichlet, load_client_data
from src.model import BaseRepresentation, ClientHead
from src.client import FedRepClient

# 1. ADD THIS FUNCTION to calculate the weighted average of client accuracies
def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Return the aggregated accuracy
    return {"accuracy": sum(accuracies) / sum(examples)}

def main():
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Update this line:
    global_trainset, global_valset, client_train_indices, client_val_indices = prepare_dataset_dirichlet(
        config['num_clients'], config['dirichlet_alpha']
    )

    def client_fn(cid: str) -> fl.client.Client:
        # And update this line:
        trainloader, valloader = load_client_data(
            int(cid), global_trainset, global_valset, client_train_indices, client_val_indices, config['batch_size']
        )
        base_model = BaseRepresentation()
        head_model = ClientHead()
        return FedRepClient(cid, base_model, head_model, trainloader, valloader, config, device)

    # 2. UPDATE THE STRATEGY to include the evaluate_metrics_aggregation_fn
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=config['fraction_fit'], 
        fraction_evaluate=1.0,
        min_fit_clients=int(config['num_clients'] * config['fraction_fit']),
        min_evaluate_clients=config['num_clients'],
        min_available_clients=config['num_clients'],
        evaluate_metrics_aggregation_fn=weighted_average, # <--- ADD THIS LINE
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config['num_clients'],
        config=fl.server.ServerConfig(num_rounds=config['num_rounds']),
        strategy=strategy,
    )
    
    # 3. PRINT THE FINAL ACCURACY to the terminal so you can record it!
    print("\n=== FINAL RESULTS ===")
    print(f"Final Aggregated Accuracy: {history.metrics_distributed['accuracy'][-1][1] * 100:.2f}%")

if __name__ == "__main__":
    main()