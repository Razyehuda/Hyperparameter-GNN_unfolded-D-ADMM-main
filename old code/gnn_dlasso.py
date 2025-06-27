import gnn_dlasso_utils
import gnn_dlasso_models2

from configurations import args_parser
import torch
import gnn_data
import networkx as nx
import os
import pandas as pd
from datetime import datetime
import csv
from tqdm import tqdm

# Add mixed precision training
from torch.cuda.amp import GradScaler, autocast

if __name__ == "__main__":
    args = args_parser()
    A = gnn_dlasso_utils.set_A(args)

    train_loader = gnn_data.set_Data(A, data_len=args.train_size, args=args)
    valid_loader = gnn_data.set_Data(A, data_len=args.test_size, args=args)

    model = gnn_dlasso_models2.DLASSO_GNNHyp2(A=A, args=args)
    
    # Move model to device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    A = A.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)
    
    # Add gradient clipping
    max_grad_norm = 1.0
    
    # Better learning rate scheduler with warmup
    def get_lr_scheduler(optimizer, num_warmup_steps=100, num_training_steps=1000):
        from torch.optim.lr_scheduler import OneCycleLR
        return OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=num_training_steps,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',
            cycle_momentum=False
        )
    
    # Calculate total training steps
    total_steps = args.num_epochs * args.GHN_iter_num * (args.train_size // args.batch_size)
    scheduler = get_lr_scheduler(optimizer, num_training_steps=total_steps)
    
    # Early stopping
    best_valid_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Model checkpointing
    checkpoint_dir = os.path.join("checkpoints", "gnn_dlasso")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None

    training_losses = {'mean': [], 'final': []}
    validation_losses = {'mean': [], 'final': []}

    layer_training_loss = []
    layer_valid_loss = []
    for k in range(1,args.GHN_iter_num+1):
        training_losses = {'mean': [], 'final': []}
        validation_losses = {'mean': [], 'final': []}
        print(f'\nLayer: {k} |')
        for epoch in range(args.num_epochs):
            print(f'\nepoch: {epoch + 1} |')

            train_loss_mean = 0.0
            train_loss_final = 0.0
            for iter, (b, label) in enumerate(tqdm(train_loader, desc=f"Layer {k}, Epoch {epoch + 1}", leave=False)):
                model.train()
                batch_size = b.shape[0]
                
                # Move data to device
                b = b.to(device)
                label = label.to(device)
                
                graph_list = [nx.erdos_renyi_graph(args.P, args.graph_prob) for _ in range(batch_size)]
                
                # Mixed precision training
                if scaler is not None:
                    with autocast():
                        Y = model(b, graph_list)
                        loss_mean, loss_final = gnn_dlasso_utils.compute_loss(Y, label)
                        loss = (loss_mean + loss_final) / 2
                    
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    Y = model(b, graph_list)
                    loss_mean, loss_final = gnn_dlasso_utils.compute_loss(Y, label)
                    loss = (loss_mean + loss_final) / 2

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                train_loss_mean += loss_mean.item()
                train_loss_final += loss_final.item()

            train_loss_mean /= len(train_loader)
            train_loss_final /= len(train_loader)

            training_losses['mean'].append(train_loss_mean)
            training_losses['final'].append(train_loss_final)

            print(f'train loss  (mean): {train_loss_mean:.5f}')
            print(f'train loss (final): {train_loss_final:.5f}')
            print(f'learning rate: {optimizer.param_groups[0]["lr"]:.6f}')

            # Validation
            with torch.no_grad():
                valid_loss_mean = 0.0
                valid_loss_final = 0.0
                for iter, (b, label) in enumerate(tqdm(valid_loader, desc=f"Layer {k}, Validation", leave=False)):
                    model.eval()
                    batch_size = b.shape[0]
                    
                    # Move data to device
                    b = b.to(device)
                    label = label.to(device)
                    
                    graph_list = [nx.erdos_renyi_graph(args.P, args.graph_prob) for _ in range(batch_size)]
                    Y = model(b, graph_list)#, K = k) # K = k
                    loss_mean, loss_final = gnn_dlasso_utils.compute_loss(Y, label)

                    valid_loss_mean += loss_mean.item()
                    valid_loss_final += loss_final.item()

                valid_loss_mean /= len(valid_loader)
                valid_loss_final /= len(valid_loader)

                validation_losses['mean'].append(valid_loss_mean)
                validation_losses['final'].append(valid_loss_final)

                print(f'valid loss  (mean): {valid_loss_mean:.5f}')
                print(f'valid loss (final): {valid_loss_final:.5f}')
                
                # Step the scheduler based on validation loss
                scheduler.step(valid_loss_final)
                
                # Early stopping logic
                if valid_loss_final < best_valid_loss:
                    best_valid_loss = valid_loss_final
                    patience_counter = 0
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'valid_loss': valid_loss_final,
                        'layer': k
                    }, os.path.join(checkpoint_dir, f'best_model_layer_{k}.pt'))
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1} for layer {k}")
                    break
                
        layer_training_loss.append(training_losses)
        layer_valid_loss.append(validation_losses)


    # ===== Save Outputs to Timestamped Directory (once after all training) =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))  # path to this script
    save_dir = os.path.join(base_dir, "results", f"{timestamp}_GNN")

    os.makedirs(save_dir, exist_ok=True)

    # Save all training and validation losses (per layer) as CSVs
    for i, (train, valid) in enumerate(zip(layer_training_loss, layer_valid_loss)):
        train_df = pd.DataFrame(train)
        valid_df = pd.DataFrame(valid)
        train_df.to_csv(os.path.join(save_dir, f'train_losses_layer_{i+1}.csv'), index=False)
        valid_df.to_csv(os.path.join(save_dir, f'valid_losses_layer_{i+1}.csv'), index=False)

    # Save args (as a pickled object)
    torch.save(args, os.path.join(save_dir, "args.pt"))

    # Save A matrix
    torch.save(A, os.path.join(save_dir, "A.pt"))

    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    print(f"\n✅ All results saved to '{save_dir}'")
