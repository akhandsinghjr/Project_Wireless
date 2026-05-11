import re
import matplotlib.pyplot as plt
import os

def parse_results(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Split into FedRep and FedAvg blocks (assuming FedAvg is marked by /// Fedavg)
    parts = content.split('/// Fedavg')
    fedrep_text = parts[0]
    fedavg_text = parts[1] if len(parts) > 1 else ""

    def extract_metrics(text):
        experiments = {}
        # Split by Dirichlet = ...
        blocks = re.split(r'Dirichlet\s*=\s*(.+)', text)[1:]
        
        for i in range(0, len(blocks), 2):
            alpha = blocks[i].strip().replace("'", "")
            data = blocks[i+1]
            
            # Extract loss
            loss_matches = re.findall(r'round \d+:\s+([\d\.]+)', data)
            losses = [float(l) for l in loss_matches]
            
            # Extract accuracy
            acc_matches = re.findall(r'\(\d+,\s*([\d\.]+)\)', data)
            accuracies = [float(a) * 100 for a in acc_matches] # Convert to percentage
            
            if losses and accuracies:
                experiments[alpha] = {'loss': losses, 'accuracy': accuracies}
        return experiments

    return extract_metrics(fedrep_text), extract_metrics(fedavg_text)

def generate_plots():
    os.makedirs('results', exist_ok=True)
    fedrep_data, fedavg_data = parse_results('results.txt')

    rounds = list(range(1, 51))

    # ---------------------------------------------------------
    # PLOT 1: Loss vs Communication Rounds (Extreme Heterogeneity)
    # Rubric Requirement: "Global loss vs. communication rounds"
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=300)
    
    # We plot alpha = 0.01 for both to show how FedAvg explodes and FedRep converges
    plt.plot(rounds, fedavg_data['0.01']['loss'], label='FedAvg ($\\alpha=0.01$)', color='red', linestyle='--', alpha=0.7)
    plt.plot(rounds, fedrep_data['0.01']['loss'], label='FedRep ($\\alpha=0.01$)', color='green', linewidth=2)
    
    plt.title('Training Loss vs. Communication Rounds (Extreme Non-IID)', fontsize=14)
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/loss_comparison_alpha_0.01.png', bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: FedRep Accuracy Across Different Data Distributions
    # Rubric Requirement: "Global accuracy vs. rounds (all experiments)"
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=300)
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    alphas_to_plot = ['0.01', '0.1', '0.5', '1.0', 'IID']
    
    for idx, alpha in enumerate(alphas_to_plot):
        if alpha in fedrep_data:
            plt.plot(rounds, fedrep_data[alpha]['accuracy'], label=f'FedRep ($\\alpha={alpha}$)', color=colors[idx], linewidth=1.5)

    plt.title('FedRep Personalized Accuracy Across Data Distributions', fontsize=14)
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/fedrep_all_alphas_accuracy.png', bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # PLOT 3: The FedAvg vs FedRep "Flip" (IID vs Non-IID)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # Left Plot: IID Data
    ax1.plot(rounds, fedavg_data['IID']['accuracy'], label='FedAvg', linestyle='--', color='red')
    ax1.plot(rounds, fedrep_data['IID']['accuracy'], label='FedRep', color='green')
    ax1.set_title('IID Data (Perfectly Uniform)', fontsize=13)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right Plot: 0.01 Data
    ax2.plot(rounds, fedavg_data['0.01']['accuracy'], label='FedAvg', linestyle='--', color='red')
    ax2.plot(rounds, fedrep_data['0.01']['accuracy'], label='FedRep', color='green')
    ax2.set_title('Non-IID Data ($\\alpha=0.01$)', fontsize=13)
    ax2.set_xlabel('Round')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Algorithm Performance: IID vs Extreme Non-IID', fontsize=16, y=1.05)
    plt.savefig('results/iid_vs_non_iid_trajectories.png', bbox_inches='tight')
    plt.close()

    print("Successfully generated 3 new plots in the /results folder!")

if __name__ == "__main__":
    generate_plots()