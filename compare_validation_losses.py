import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

def load_validation_losses():
    """
    Load validation losses from the three different CSV files
    """
    # Define paths
    csv_folder1_path = "results/csv_folder1/losses.csv"  # unfolded_dlasso on single graph
    csv_folder2_path = "results/csv_folder2/losses.csv"  # unfolded_dlasso on multiple graphs  
    best_model_path = "checkpoints/best_our_model_25iter/valid_losses.csv"  # updated path
    
    # Load data
    try:
        df1 = pd.read_csv(csv_folder1_path)
        print(f"✅ Loaded {len(df1)} epochs from csv_folder1 (single graph)")
    except FileNotFoundError:
        print(f"❌ File not found: {csv_folder1_path}")
        df1 = None
    
    try:
        df2 = pd.read_csv(csv_folder2_path)
        print(f"✅ Loaded {len(df2)} epochs from csv_folder2 (multiple graphs)")
    except FileNotFoundError:
        print(f"❌ File not found: {csv_folder2_path}")
        df2 = None
    
    try:
        df_best = pd.read_csv(best_model_path)
        print(f"✅ Loaded {len(df_best)} epochs from best_our_model_25iter")
    except FileNotFoundError:
        print(f"❌ File not found: {best_model_path}")
        df_best = None
    
    return df1, df2, df_best

def analyze_validation_losses(df1, df2, df_best):
    """
    Analyze and compare validation losses
    """
    results = {}
    
    # Analyze each dataset
    datasets = [
        ("Single Graph (Unfolded DLASSO)", df1),
        ("Multiple Graphs (Unfolded DLASSO)", df2),
        ("Multiple Graphs (GNN DLASSO)", df_best)
    ]
    
    for name, df in datasets:
        if df is not None:
            # Always use the rightmost column for validation loss
            valid_losses = df.iloc[:, -1].values
            final_loss = valid_losses[-1]
            threshold = final_loss * 0.01  # 1% threshold
            convergence_epoch = None
            
            for i, loss in enumerate(valid_losses):
                if abs(loss - final_loss) <= threshold:
                    convergence_epoch = i + 1
                    break
            
            # Stability: std of last 10 epochs
            if len(valid_losses) >= 10:
                stability_std = np.std(valid_losses[-10:])
            else:
                stability_std = np.std(valid_losses)
            
            results[name] = {
                'final_loss': final_loss,
                'convergence_epoch': convergence_epoch,
                'all_losses': valid_losses,
                'stability_std_last10': stability_std
            }
    
    return results

def print_and_save_summary(results, save_dir):
    """
    Print and save a summary table
    """
    os.makedirs(save_dir, exist_ok=True)
    table_data = []
    for name, data in results.items():
        table_data.append({
            'Model': name,
            'Final Loss': f"{data['final_loss']:.6f}",
            'Convergence Epoch': data['convergence_epoch'] if data['convergence_epoch'] else 'N/A',
            'Stability (Std Last 10)': f"{data['stability_std_last10']:.6f}",
            'Total Epochs': len(data['all_losses'])
        })
    df_summary = pd.DataFrame(table_data)
    df_summary.to_csv(os.path.join(save_dir, 'validation_loss_summary.csv'), index=False)
    print("\n" + "="*60)
    print("VALIDATION LOSS SUMMARY")
    print("="*60)
    print(df_summary.to_string(index=False))
    print("="*60)
    # Rankings
    best_final = min(table_data, key=lambda x: float(x['Final Loss']))
    best_converge = min([d for d in table_data if d['Convergence Epoch'] != 'N/A'], key=lambda x: int(x['Convergence Epoch']))
    best_stable = min(table_data, key=lambda x: float(x['Stability (Std Last 10)']))
    print(f"\n🏆 Best Final Loss: {best_final['Model']} ({best_final['Final Loss']})")
    print(f"🚀 Fastest Convergence: {best_converge['Model']} (Epoch {best_converge['Convergence Epoch']})")
    print(f"🧊 Most Stable (Lowest Std Last 10): {best_stable['Model']} ({best_stable['Stability (Std Last 10)']})")

def plot_comparisons(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    names = list(results.keys())
    # Find the best model (lowest final loss)
    best_model = min(results.items(), key=lambda x: x[1]['final_loss'])[0]
    # Assign colors: best=red, first=blue, second=green
    color_map = {}
    alt_colors = ['blue', 'green']
    alt_idx = 0
    for name in names:
        if name == best_model:
            color_map[name] = 'red'
        else:
            color_map[name] = alt_colors[alt_idx]
            alt_idx += 1
    colors = [color_map[name] for name in names]
    # 1. Validation loss curves
    plt.figure(figsize=(10,6))
    for i, (name, data) in enumerate(results.items()):
        plt.plot(range(1, len(data['all_losses'])+1), data['all_losses'], 
                label=name, linewidth=2, color=colors[i])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'validation_loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    # 2. Bar chart: final loss
    plt.figure(figsize=(8,5))
    final_losses = [data['final_loss'] for data in results.values()]
    bars = plt.bar(names, final_losses, color=colors, alpha=0.7)
    plt.ylabel('Final Validation Loss')
    plt.title('Final Validation Loss Comparison')
    plt.xticks(rotation=30, ha='right')
    for i, v in enumerate(final_losses):
        plt.text(i, v+0.001, f'{v:.4f}', ha='center', fontweight='bold')
    plt.tight_layout(rect=[0,0.15,1,1])
    plt.savefig(os.path.join(save_dir, 'final_loss_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    # 3. Bar chart: convergence epoch
    plt.figure(figsize=(8,5))
    conv_epochs = [data['convergence_epoch'] if data['convergence_epoch'] else 0 for data in results.values()]
    bars = plt.bar(names, conv_epochs, color=colors, alpha=0.7)
    plt.ylabel('Convergence Epoch')
    plt.title('Convergence Epoch Comparison')
    plt.xticks(rotation=30, ha='right')
    for i, v in enumerate(conv_epochs):
        plt.text(i, v+0.5, f'{v}', ha='center', fontweight='bold')
    plt.tight_layout(rect=[0,0.15,1,1])
    plt.savefig(os.path.join(save_dir, 'convergence_epoch_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    # 4. Bar chart: stability (std last 10)
    plt.figure(figsize=(8,5))
    stabilities = [data['stability_std_last10'] for data in results.values()]
    bars = plt.bar(names, stabilities, color=colors, alpha=0.7)
    plt.ylabel('Std of Last 10 Validation Losses')
    plt.title('Stability Comparison (Lower is Better)')
    plt.xticks(rotation=30, ha='right')
    for i, v in enumerate(stabilities):
        plt.text(i, v+0.0005, f'{v:.5f}', ha='center', fontweight='bold')
    plt.tight_layout(rect=[0,0.15,1,1])
    plt.savefig(os.path.join(save_dir, 'stability_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to run the comparison
    """
    print("🔍 Loading validation losses from CSV files...")
    
    # Load data
    df1, df2, df_best = load_validation_losses()
    
    if all(df is None for df in [df1, df2, df_best]):
        print("❌ No CSV files found. Please check the file paths.")
        return
    
    # Analyze results
    print("\n📊 Analyzing validation losses...")
    results = analyze_validation_losses(df1, df2, df_best)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/validation_loss_summary_{timestamp}"
    
    # Print and save summary
    print(f"\n📈 Printing and saving summary...")
    print_and_save_summary(results, save_dir)
    
    plot_comparisons(results, save_dir)
    
    print(f"\n✅ Summary and plots saved to: {save_dir}")

if __name__ == "__main__":
    main() 