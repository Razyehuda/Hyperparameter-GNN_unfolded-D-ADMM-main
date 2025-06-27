import gnn_dlasso_utils
import gnn_dlasso_models_progressive

from configurations import args_parser
import torch
import gnn_data
import networkx as nx
import os
import pandas as pd
from datetime import datetime
import csv
from tqdm import tqdm
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt

# Add mixed precision training
from torch.cuda.amp import GradScaler, autocast


if __name__ == "__main__":
    args = args_parser()
    A = gnn_dlasso_utils.set_A(args)

    train_loader = gnn_data.set_Data(A, data_len=args.train_size, args=args)
    valid_loader = gnn_data.set_Data(A, data_len=args.test_size, args=args)

    model = gnn_dlasso_models_progressive.DLASSO_GNNHyp3_Progressive(A=A, args=args)
    
    # Move model to device - force CUDA if specified
    if 'cuda' in args.device.lower():
        device = torch.device(args.device)
        print(f"Forcing CUDA device: {device}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    A = A.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # Add gradient clipping - increased for more learning freedom
    max_grad_norm = 100.0
    
    # More stable learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=15, min_lr=1e-6
    )
    
    # Early stopping
    best_valid_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # Model checkpointing - NEW DIRECTORY STRUCTURE
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", f"progressive_model_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Mixed precision training - disable for better performance on smaller models
    use_mixed_precision = False  # Set to True if you want mixed precision
    scaler = GradScaler() if device.type == 'cuda' and use_mixed_precision else None
    
    training_losses = {'mean': [], 'final': []}
    validation_losses = {'mean': [], 'final': []}

    # Continuous progressive learning
    # Start with 2 iterations, gradually increase to 15
    min_iterations = 1  # Start with 2 iterations for meaningful learning
    max_iterations = args.GHN_iter_num  # 15
    total_epochs = args.num_epochs
    
    # Calculate iterations per epoch (more epochs at lower iterations)
    # This gives a smooth progression from 2 to 15 iterations
    def get_iterations_for_epoch(epoch):
        # Use exponential growth for smooth progression
        # Reach max iterations at 75% of epochs, then stay at max
        progress = epoch / (total_epochs * 0.75)  # Scale to 75% of total epochs
        progress = min(1.0, progress)  # Cap at 1.0 (max iterations)
        iterations = min_iterations + (max_iterations - min_iterations) * (progress ** 1.5)
        return max(min_iterations, min(max_iterations, round(iterations)))
    
    def adjust_learning_rate_for_iterations(optimizer, current_iterations, epoch, total_epochs, base_lr):
        """
        Adjust learning rate only when at maximum iterations for fine-tuning
        - Only reduce LR when stuck at max iterations
        - Keep original LR for all other iteration counts
        """
        # Only reduce learning rate when at maximum iterations
        if current_iterations >= max_iterations:
            # Calculate how long we've been at max iterations
            max_iter_epoch = int(total_epochs * 0.75)  # Epoch when we first reach max iterations
            epochs_at_max = epoch - max_iter_epoch + 1
            
            # Reduce LR based on how long we've been at max iterations
            # Start with 0.8 of original LR, reduce to 0.3 over the remaining epochs
            remaining_epochs = total_epochs - max_iter_epoch
            if remaining_epochs > 0:
                reduction_factor = 0.8 - (epochs_at_max / remaining_epochs) * 0.5
                reduction_factor = max(0.3, reduction_factor)  # Minimum 0.3 of original LR
            else:
                reduction_factor = 0.8
            
            # Apply to all parameter groups
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * reduction_factor
            
            return reduction_factor
        else:
            # Keep original learning rate for all other iteration counts
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr
            
            return 1.0

    def plot_iteration_progression(total_epochs, checkpoint_dir):
        """Plot the progression of iterations over epochs"""
        epochs = list(range(total_epochs))
        iterations = [get_iterations_for_epoch(epoch) for epoch in epochs]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, iterations, 'o-', color='green', linewidth=2, markersize=4)
        plt.title('Progressive Learning: Iteration Progression')
        plt.xlabel('Epoch')
        plt.ylabel('Number of Iterations')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max_iterations + 1)
        
        # Add some milestone annotations
        milestone_epochs = [0, total_epochs//4, total_epochs//2, 3*total_epochs//4, total_epochs-1]
        for epoch in milestone_epochs:
            if epoch < total_epochs:
                iter_count = get_iterations_for_epoch(epoch)
                plt.annotate(f'{iter_count} iter', 
                           xy=(epoch, iter_count), 
                           xytext=(10, 10), 
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, 'iteration_progression.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Starting progressive training with {args.num_epochs} epochs...")
    print(f"Model will be saved to: {checkpoint_dir}")
    print(f"Requested device: {args.device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"Using device: {device}")
    print(f"Mixed precision: {use_mixed_precision}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Progressive learning: {min_iterations} → {max_iterations} iterations")
    
    for epoch in range(args.num_epochs):
        current_iterations = get_iterations_for_epoch(epoch)
        
        # Adjust learning rate based on iteration progression
        lr_factor = adjust_learning_rate_for_iterations(optimizer, current_iterations, epoch, total_epochs, args.lr)
        
        print(f'\nEpoch: {epoch + 1}/{args.num_epochs} | Iterations: {current_iterations} | LR Factor: {lr_factor:.3f}')

        # Training phase
        model.train()
        train_loss_mean = 0.0
        train_loss_final = 0.0
        
        for iter, (b, label) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)):
            batch_size = b.shape[0]
            
            # Move data to device
            b = b.to(device)
            label = label.to(device)
            
            # Generate simple random graphs for each sample in the batch
            graph_list = []
            for _ in range(batch_size):
                # Ensure minimum connectivity for better message passing
                graph = nx.erdos_renyi_graph(args.P, max(args.graph_prob, 0.3))
                # Ensure graph is connected
                if not nx.is_connected(graph):
                    # Add edges to make it connected
                    components = list(nx.connected_components(graph))
                    for i in range(len(components) - 1):
                        graph.add_edge(list(components[i])[0], list(components[i+1])[0])
                graph_list.append(graph)
            
            # Mixed precision training
            if scaler is not None:
                with autocast():
                    Y, (alpha_k, tau_k, rho_k, eta_k) = model(b, graph_list, training_iterations=current_iterations)
                    loss_mean, loss_final = gnn_dlasso_utils.compute_loss(Y, label)
                    loss = loss_final
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                Y, (alpha_k, tau_k, rho_k, eta_k) = model(b, graph_list, training_iterations=current_iterations)
                loss_mean, loss_final = gnn_dlasso_utils.compute_loss(Y, label)
                loss = loss_final
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            # Print hyperparameters for the first sample (only every 10 batches to avoid spam)
            if iter % 10 == 0:
                with torch.no_grad():
                    # Get hyperparameters for the first sample
                    alpha_val = alpha_k[0, 0].item()
                    tau_val = tau_k[0, 0].item()
                    rho_val = rho_k[0, 0].item()
                    eta_val = eta_k[0, 0].item()
                    print(f"Batch {iter}: Alpha: {alpha_val:.6f}, Tau: {tau_val:.6f}, Rho: {rho_val:.6f}, Eta: {eta_val:.6f}")

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
                
                # Generate simple random graphs for each sample in the batch
                graph_list = []
                for _ in range(batch_size):
                    # Ensure minimum connectivity for better message passing
                    graph = nx.erdos_renyi_graph(args.P, max(args.graph_prob, 0.3))
                    # Ensure graph is connected
                    if not nx.is_connected(graph):
                        # Add edges to make it connected
                        components = list(nx.connected_components(graph))
                        for i in range(len(components) - 1):
                            graph.add_edge(list(components[i])[0], list(components[i+1])[0])
                    graph_list.append(graph)
                
                Y, _ = model(b, graph_list, training_iterations=current_iterations)
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
                    'args': args,
                    'current_iterations': current_iterations
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
                print(f"✅ New best model saved! Loss: {valid_loss_final:.5f} (Iterations: {current_iterations})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break

    # Save final results
    print(f"\n📊 Progressive training completed! Saving results to {checkpoint_dir}")
    
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
        'args': args,
        'final_iterations': current_iterations
    }, os.path.join(checkpoint_dir, 'final_model.pt'))

    # Save args and A matrix
    torch.save(args, os.path.join(checkpoint_dir, "args.pt"))
    torch.save(A, os.path.join(checkpoint_dir, "A.pt"))

    print(f"✅ All results saved to '{checkpoint_dir}'")
    print(f"🎯 Best validation loss: {best_valid_loss:.5f}")
    print(f"📈 Final validation loss: {valid_loss_final:.5f}")
    print(f"🔄 Final iterations: {current_iterations}")

    # Plot training and validation losses
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Training vs Validation Loss (final)
    plt.subplot(1, 2, 1)
    plt.plot(training_losses['final'], label='Training Loss (final)', color='blue', linewidth=2)
    plt.plot(validation_losses['final'], label='Validation Loss (final)', color='red', linewidth=2)
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Iteration progression
    epochs = list(range(len(training_losses['final'])))
    iterations = [get_iterations_for_epoch(epoch) for epoch in epochs]
    plt.subplot(1, 2, 2)
    plt.plot(epochs, iterations, 'o-', color='green', linewidth=2, markersize=4)
    plt.title('Iteration Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Iterations')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max_iterations + 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'training_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot iteration progression
    plot_iteration_progression(total_epochs, checkpoint_dir) 