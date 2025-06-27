import gnn_dlasso_utils
import gnn_dlasso_models_new
import utils
import tqdm
from configurations import args_parser
import torch
import gnn_data
import networkx as nx
import os
import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt


if __name__ == "__main__":
    args = args_parser()
    print(f"alpha_max from args: {args.alpha_max}")
    
    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    print(f"Using device: {device}")
    
    A = gnn_dlasso_utils.set_A(args)
    A = A.to(device)  # Move A tensor to device

    train_loader = gnn_data.set_Data(A, data_len=args.train_size, args=args)
    valid_loader = gnn_data.set_Data(A, data_len=args.test_size, args=args)

    model = gnn_dlasso_models_new.DLASSO_GNNHypNew(A=A, args=args)
    model = model.to(device)  # Move model to device

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=False, weight_decay=1e-4)

    graph = nx.erdos_renyi_graph(args.P, args.graph_prob)

    # Training and validation losses tracking
    training_losses = []
    validation_losses = []
    
    for epoch in range(args.num_epochs):
        print(f'\nEpoch: {epoch + 1}/{args.num_epochs}')
        
        train_loss = 0.0
        
        # Training with tqdm progress bar
        train_pbar = tqdm.tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for iter, (b, label) in enumerate(train_pbar):
            batch_size = b.shape[0]
            graph_list = [graph for _ in range(batch_size)]
            
            # Move data to device
            b = b.to(device)
            label = label.to(device)
            
            # Call model - it will use the full number of iterations
            Y, hyp = model(b, graph_list)
            loss_mean, loss_final = gnn_dlasso_utils.compute_loss(Y, label)
            loss = loss_final

            optimizer.zero_grad()
            loss.backward()
            
            # Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            
            optimizer.step()

            train_loss += loss_final.item()
            
            # Update progress bar
            train_pbar.set_postfix({'Loss': f'{loss_final.item():.5f}'})

        train_loss /= len(train_loader)
        training_losses.append(train_loss)

        print(f'Train Loss: {train_loss:.5f}')

        # Print mean hyperparameter values for the first batch
        with torch.no_grad():
            # Print mean hyperparameter values for the first batch (RAW and SCALED)
            alpha_raw = hyp[0, 0].mean().item()
            tau_raw   = hyp[0, 1].mean().item()
            rho_raw   = hyp[0, 2].mean().item()
            eta_raw   = hyp[0, 3].mean().item()
            # These are the scaled values
            alpha_val = alpha_raw * model.alpha_max.item()
            tau_val   = tau_raw * model.tau_max.item()
            rho_val   = rho_raw * model.rho_max.item()
            eta_val   = eta_raw * model.eta_max.item()
            print(f'Hyperparameters (mean, SCALED) - Alpha: {alpha_val:.6f}, Tau: {tau_val:.6f}, Rho: {rho_val:.6f}, Eta: {eta_val:.6f}')
            print(f'Hyperparameters (mean, RAW)    - Alpha: {alpha_raw:.6f}, Tau: {tau_raw:.6f}, Rho: {rho_raw:.6f}, Eta: {eta_raw:.6f}')

        # Validation with tqdm progress bar
        with torch.no_grad():
            valid_loss = 0.0
            valid_pbar = tqdm.tqdm(valid_loader, desc=f'Validation Epoch {epoch+1}')
            for iter, (b, label) in enumerate(valid_pbar):
                batch_size = b.shape[0]
                graph_list = [graph for _ in range(batch_size)]
                
                # Move data to device
                b = b.to(device)
                label = label.to(device)
                
                # Call model
                Y, hyp = model(b, graph_list)
                loss_mean, loss_final = gnn_dlasso_utils.compute_loss(Y, label)

                valid_loss += loss_final.item()
                
                # Update progress bar
                valid_pbar.set_postfix({'Loss': f'{loss_final.item():.5f}'})

            valid_loss /= len(valid_loader) if len(valid_loader) > 0 else 1
            validation_losses.append(valid_loss)

            print(f'Valid Loss: {valid_loss:.5f}')

    # ===== Save Outputs to Timestamped Directory =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))  # path to this script
    save_dir = os.path.join(base_dir, "results", f"{timestamp}_gnn_new")

    os.makedirs(save_dir, exist_ok=True)

    # Save training and validation losses as CSVs
    loss_data = {
        'epoch': list(range(1, len(training_losses) + 1)),
        'train_loss': training_losses,
        'valid_loss': validation_losses
    }
    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv(os.path.join(save_dir, 'losses.csv'), index=False)

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Train Loss')
    plt.plot(validation_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'losses.png'))
    plt.close()

    # Save args (as a pickled object)
    torch.save(args, os.path.join(save_dir, "args.pt"))

    # Save A matrix
    torch.save(A, os.path.join(save_dir, "A.pt"))

    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    print(f"\n✅ All results saved to '{save_dir}'") 