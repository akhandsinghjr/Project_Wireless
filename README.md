# Federated Learning on FashionMNIST

This repository contains a Federated Learning project implemented in Python, designed to train models collaboratively on the FashionMNIST dataset using algorithms such as Federated Averaging (FedAvg).

# Video Link

Youtube : https://youtu.be/I_-reaMqa4M
# Team 

## Group - 22

Akhand Pratap Singh - 2201CS11

Anchal Dubey - 2201CS15

Aditya Chauhan - 2201CS07

Aditya Yadav - 2201CS08

Mayur Borse - 2201CS45

Saksham Singh - 2201AI33

## Project Structure

```
.
├── configs/
│   └── config.yaml          # Configuration settings for federated learning
├── data/                    # Contains the FashionMNIST dataset
├── results/                 # Directory for storing output plots and evaluation results
├── src/
│   ├── advance_plot.py      # Advanced plotting utilities
│   ├── client_fedavg.py     # Client implementation specifically for FedAvg
│   ├── client.py            # Standard client implementation
│   ├── data.py              # Data loading and preprocessing pipelines
│   ├── model.py             # Neural network model architecture
│   ├── plot_results.py      # Basic result visualization
│   └── server.py            # Federated learning server implementation
├── requirements.txt         # Python dependencies
└── results.txt              # Generated text outputs and metrics
```

## Setup & Installation

1. Create a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start a federated learning experiment, you generally need to start the central server and then connect the designated number of clients.

1. **Start the server:**
   ```bash
   python src/server.py
   ```

2. **Start the clients (in separate terminals):**
   ```bash
   python src/client.py
   ```
   *(Note: You may need to run this multiple times depending on your `config.yaml` settings or use a script to spawn multiple clients).*

3. **View Results:**
   After training completes, use the plotting scripts to visualize the performance:
   ```bash
   python src/plot_results.py
   # or
   python src/advance_plot.py
   ```
