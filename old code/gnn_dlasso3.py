import gnn_dlasso_utils
import gnn_dlasso_models3

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

    model = gnn_dlasso_models3.DLASSO_GNNHyp3(A=A, args=args)
    
    # Move model to device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    A = A.to(device)

    # Enhanced optimizer with better parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4,  # Add weight decay for regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Enhanced gradient clipping
    max_grad_norm = 10.0  # Much higher threshold for more freedom
    
    # Better learning rate scheduler with warmup
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, StepLR, ReduceLROnPlateau
    
    # Option 1: Simple StepLR (recommended for stable training)
    scheduler = StepLR(
        optimizer, 
        step_size=20,  # Reduce LR every 20 epochs
        gamma=0.8      # Multiply LR by 0.8
    )
    
    # Option 2: Improved CosineAnnealingWarmRestarts (less aggressive)
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer, 
    #     T_0=30,        # Restart every 30 epochs (less frequent)
    #     T_mult=1,      # Keep same interval (no doubling)
    #     eta_min=1e-5   # Higher minimum LR
    # )
    
    # Option 3: ReduceLROnPlateau (reduces LR when loss plateaus)
    # scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=10,
    #     verbose=True
    # )
    
    # Option 4: OneCycleLR for better convergence (uncomment to use)
    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=args.lr,
    #     epochs=args.num_epochs,
    #     steps_per_epoch=len(train_loader),
    #     pct_start=0.3,  # Warm up for 30% of training
    #     anneal_strategy='cos'
    # )
    
    # Early stopping with better patience
    best_valid_loss = float('inf')
    patience = 15  # Increased patience
    patience_counter = 0
    
    # Model checkpointing - NEW DIRECTORY STRUCTURE
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", f"new_model_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None

    training_losses = {'mean': [], 'final': []}
    validation_losses = {'mean': [], 'final': []}

    print(f"Starting training with {args.num_epochs} epochs...")
    print(f"Model will be saved to: {checkpoint_dir}")
    
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
            
            # Generate graphs with better connectivity
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
                    Y = model(b, graph_list)
                    loss_mean, loss_final = gnn_dlasso_utils.compute_loss(Y, label)
                    loss = loss_final
                    
                    # Add regularization loss
                    l2_reg = 0.0
                    for param in model.parameters():
                        l2_reg += torch.norm(param, p=2)
                    loss += 1e-5 * l2_reg  # Small L2 regularization
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                Y = model(b, graph_list)
                loss_mean, loss_final = gnn_dlasso_utils.compute_loss(Y, label)
                loss = loss_final
                
                # Add regularization loss
                l2_reg = 0.0
                for param in model.parameters():
                    l2_reg += torch.norm(param, p=2)
                loss += 1e-5 * l2_reg  # Small L2 regularization

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            # Step scheduler for OneCycleLR (if using)
            # if isinstance(scheduler, OneCycleLR):
            #     scheduler.step()

            train_loss_mean += loss_mean.item()
            train_loss_final += loss_final.item()

        train_loss_mean /= len(train_loader)
        train_loss_final /= len(train_loader)

        training_losses['mean'].append(train_loss_mean)
        training_losses['final'].append(train_loss_final)

        print(f'train loss  (mean): {train_loss_mean:.5f}')
        print(f'train loss (final): {train_loss_final:.5f}')
        print(f'learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Add scheduler info
        if isinstance(scheduler, StepLR):
            print(f'scheduler: StepLR (step {scheduler.last_epoch + 1})')
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            print(f'scheduler: CosineAnnealingWarmRestarts (T_0={scheduler.T_0}, T_mult={scheduler.T_mult})')
        elif isinstance(scheduler, ReduceLROnPlateau):
            print(f'scheduler: ReduceLROnPlateau (patience={scheduler.patience})')
        elif isinstance(scheduler, OneCycleLR):
            print(f'scheduler: OneCycleLR')
        else:
            print(f'scheduler: {type(scheduler).__name__}')

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
                
                # Use same graph generation strategy for validation
                graph_list = []
                for _ in range(batch_size):
                    graph = nx.erdos_renyi_graph(args.P, max(args.graph_prob, 0.3))
                    if not nx.is_connected(graph):
                        components = list(nx.connected_components(graph))
                        for i in range(len(components) - 1):
                            graph.add_edge(list(components[i])[0], list(components[i+1])[0])
                    graph_list.append(graph)
                
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
            
            # Step the scheduler based on type
            if isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()
            elif isinstance(scheduler, StepLR):
                scheduler.step()
            elif isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(valid_loss_final)
            # OneCycleLR is stepped per batch, not per epoch
            
            # Early stopping logic
            if valid_loss_final < best_valid_loss:
                best_valid_loss = valid_loss_final
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'valid_loss': valid_loss_final,
                    'args': args
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
                print(f"✅ New best model saved! Loss: {valid_loss_final:.5f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break

    # Save final results
    print(f"\n📊 Training completed! Saving results to {checkpoint_dir}")
    
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
        'scheduler_state_dict': scheduler.state_dict(),
        'final_valid_loss': valid_loss_final,
        'args': args
    }, os.path.join(checkpoint_dir, 'final_model.pt'))

    # Save args and A matrix
    torch.save(args, os.path.join(checkpoint_dir, "args.pt"))
    torch.save(A, os.path.join(checkpoint_dir, "A.pt"))

    print(f"✅ All results saved to '{checkpoint_dir}'")
    print(f"🎯 Best validation loss: {best_valid_loss:.5f}")
    print(f"📈 Final validation loss: {valid_loss_final:.5f}") 