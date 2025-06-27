import gnn_dlasso_utils
import gnn_dlasso_models4

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

    # Use the MPNN-based model
    model = gnn_dlasso_models4.DLASSO_GNNHyp4(A=A, args=args)
    
    # Move model to device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    A = A.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)
    
    # Add gradient clipping
    max_grad_norm = 1.0
    
    # Simple learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    best_valid_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Model checkpointing - MPNN MODEL DIRECTORY
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", f"mpnn_model_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None

    training_losses = {'mean': [], 'final': []}
    validation_losses = {'mean': [], 'final': []}

    print(f"🚀 Starting MPNN-based training with {args.num_epochs} epochs...")
    print(f"📁 Model will be saved to: {checkpoint_dir}")
    print(f"🔧 Using Message Passing Neural Networks")
    
    for epoch in range(args.num_epochs):
        print(f'\nEpoch: {epoch + 1}/{args.num_epochs} |')

        # Training phase
        model.train()
        train_loss_mean = 0.0
        train_loss_final = 0.0
        
        for iter, (b, label) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)):
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

        # Validation phase
        with torch.no_grad():
            model.eval()
            valid_loss_mean = 0.0
            valid_loss_final = 0.0
            
            for iter, (b, label) in enumerate(tqdm(valid_loader, desc=f"Validation Epoch {epoch + 1}", leave=False)):
                batch_size = b.shape[0]
                
                # Move data to device
                b = b.to(device)
                label = label.to(device)
                
                graph_list = [nx.erdos_renyi_graph(args.P, args.graph_prob) for _ in range(batch_size)]
                Y = model(b, graph_list)
                loss_mean, loss_final = gnn_dlasso_utils.compute_loss(Y, label)

                valid_loss_mean += loss_mean.item()
                valid_loss_final += loss_final.item()

            valid_loss_mean /= len(valid_loader)
            valid_loss_final /= len(valid_loader)

            validation_losses['mean'].append(valid_loss_mean)
            validation_losses['final'].append(valid_loss_final)

            print(f'valid loss  (mean): {valid_loss_mean:.5f}')
            print(f'valid loss (final): {valid_loss_final:.5f}')
            
            # Step the scheduler
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
                    'args': args
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
                print(f"✅ New best MPNN model saved! Loss: {valid_loss_final:.5f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break

    # Save final results
    print(f"\n📊 MPNN training completed! Saving results to {checkpoint_dir}")
    
    # Save training and validation losses
    train_df = pd.DataFrame(training_losses)
    valid_df = pd.DataFrame(validation_losses)
    train_df.to_csv(os.path.join(checkpoint_dir, 'train_losses.csv'), index=False)
    valid_df.to_csv(os.path.join(checkpoint_dir, 'valid_losses.csv'), index=False)

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_valid_loss': valid_loss_final,
        'args': args
    }, os.path.join(checkpoint_dir, 'final_model.pt'))

    # Save args and A matrix
    torch.save(args, os.path.join(checkpoint_dir, "args.pt"))
    torch.save(A, os.path.join(checkpoint_dir, "A.pt"))

    print(f"✅ All MPNN results saved to '{checkpoint_dir}'")
    print(f"🎯 Best validation loss: {best_valid_loss:.5f}")
    print(f"📈 Final validation loss: {valid_loss_final:.5f}")
    print(f"🔬 Model type: Message Passing Neural Network (MPNN)") 