import gnn_dlasso_utils
import gnn_dlasso_models
import utils
import tqdm
from configurations import args_parser
import torch
import gnn_data
import networkx as nx
import unfolded_DLASSO
import os
import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


if __name__ == "__main__":
    args = args_parser()
    
    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    print(f"Using device: {device}")
    
    A = gnn_dlasso_utils.set_A(args)
    A = A.to(device)  # Move A tensor to device

    train_loader = gnn_data.set_Data(A, data_len=args.train_size, args=args)
    valid_loader = gnn_data.set_Data(A, data_len=args.test_size, args=args)

    model = unfolded_DLASSO.DLASSO_unfolded(A=A, args=args)
    model = model.to(device)  # Move model to device

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)

    # Add learning rate scheduler for hyperparameter stability
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.8, 
        patience=3, 
        min_lr=1e-6
    )

    #graph = nx.erdos_renyi_graph(args.P, args.graph_prob)

    # Training and validation losses tracking
    training_losses = []
    validation_losses = []
    
    # Early stopping parameters
    patience = 70  # Number of epochs to wait for improvement
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    graph = nx.erdos_renyi_graph(args.P, args.graph_prob)
    for epoch in range(args.num_epochs):
        print(f'\nEpoch: {epoch + 1}/{args.num_epochs}')
        
        train_loss = 0.0
        
        # Training with tqdm progress bar
        train_pbar = tqdm.tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for iter, (b, label) in enumerate(train_pbar):
            batch_size = b.shape[0]

            graph_list = [ graph for _ in range(batch_size)]
            
            # Move data to device
            b = b.to(device)
            label = label.to(device)
            
            # Call model without K parameter - it will use the full number of iterations
            Y, hyp = model(b, graph_list)
            loss_mean, loss_final = gnn_dlasso_utils.compute_loss(Y, label)
            loss = loss_final

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss_final.item()
            
            # Update progress bar
            train_pbar.set_postfix({'Loss': f'{loss_final.item():.5f}'})

        train_loss /= len(train_loader)
        training_losses.append(train_loss)

        print(f'Train Loss: {train_loss:.5f}')

        # Print hyperparameters for this epoch
        with torch.no_grad():
            # Get hyperparameters from the last training batch
            alpha_val = hyp[0, 0].item()
            tau_val = hyp[0, 1].item()
            rho_val = hyp[0, 2].item()
            eta_val = hyp[0, 3].item()
            print(f'Hyperparameters - Alpha: {alpha_val:.6f}, Tau: {tau_val:.6f}, Rho: {rho_val:.6f}, Eta: {eta_val:.6f}')

        # Validation with tqdm progress bar
        with torch.no_grad():
            valid_loss = 0.0
            valid_pbar = tqdm.tqdm(valid_loader, desc=f'Validation Epoch {epoch+1}')
            for iter, (b, label) in enumerate(valid_pbar):
                batch_size = b.shape[0]
                graph_list = [ graph for _ in range(batch_size)]
                
                # Move data to device
                b = b.to(device)
                label = label.to(device)
                
                # Call model without K parameter
                Y, hyp = model(b, graph_list)
                loss_mean, loss_final = gnn_dlasso_utils.compute_loss(Y, label)

                valid_loss += loss_final.item()
                
                # Update progress bar
                valid_pbar.set_postfix({'Loss': f'{loss_final.item():.5f}'})

            valid_loss /= len(valid_loader) if len(valid_loader) > 0 else 1
            validation_losses.append(valid_loss)

            print(f'Valid Loss: {valid_loss:.5f}')
            
            # Step the scheduler based on validation loss
            scheduler.step(valid_loss)

        # Early stopping check
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    # Restore best model weights before saving
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ===== Save Outputs to Timestamped Directory =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))  # path to this script
    save_dir = os.path.join(base_dir, "results", f"{timestamp}_unfolded_new")

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
    plt.plot(training_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(validation_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'losses.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save args (as a pickled object)
    torch.save(args, os.path.join(save_dir, "args.pt"))

    # Save A matrix
    torch.save(A, os.path.join(save_dir, "A.pt"))

    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    print(f"\n✅ All results saved to '{save_dir}'") 