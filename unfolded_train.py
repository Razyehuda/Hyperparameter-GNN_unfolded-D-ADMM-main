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

    graph = nx.erdos_renyi_graph(args.P, args.graph_prob)


    layer_training_loss = []
    layer_valid_loss = []
    for k in range(1,args.GHN_iter_num):
        training_losses = {'mean': [], 'final': []}
        validation_losses = {'mean': [], 'final': []}
        print(f'\nLayer: {k} |')
        for epoch in range(args.num_epochs):
            print(f'\nepoch: {epoch + 1} |')

            train_loss_mean = 0.0
            train_loss_final = 0.0
            for iter, (b, label) in enumerate(train_loader):
                batch_size = b.shape[0]
                graph_list = [graph for _ in range(batch_size)]
                
                # Move data to device
                b = b.to(device)
                label = label.to(device)
                
                Y = model(b, graph_list,K=k)
                loss_mean, loss_final = gnn_dlasso_utils.compute_loss(Y, label) #changed from loss2 to loss
                loss = (loss_mean + loss_final) / 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_mean += loss_mean.item()
                train_loss_final += loss_final.item()

            train_loss_mean /= len(train_loader)
            train_loss_final /= len(train_loader)

            training_losses['mean'].append(train_loss_mean)
            training_losses['final'].append(train_loss_final)

            print(f'train loss  (mean): {train_loss_mean:.5f}')
            print(f'train loss (final): {train_loss_final:.5f}')

            # Validation
            with torch.no_grad():
                valid_loss_mean = 0.0
                valid_loss_final = 0.0
                for iter, (b, label) in enumerate(valid_loader):
                    batch_size = b.shape[0]
                    graph_list = [graph for _ in range(batch_size)]
                    
                    # Move data to device
                    b = b.to(device)
                    label = label.to(device)
                    
                    Y = model(b, graph_list,K=k)
                    loss_mean, loss_final = gnn_dlasso_utils.compute_loss(Y, label) # changed to loss from loos2

                    valid_loss_mean += loss_mean.item()
                    valid_loss_final += loss_final.item()

                valid_loss_mean /= len(valid_loader) if len(valid_loader) > 0 else 1
                valid_loss_final /= len(valid_loader) if len(valid_loader) > 0 else 1

                validation_losses['mean'].append(valid_loss_mean)
                validation_losses['final'].append(valid_loss_final)

                print(f'valid loss  (mean): {valid_loss_mean:.5f}')
                print(f'valid loss (final): {valid_loss_final:.5f}')
        layer_training_loss.append(training_losses)
        layer_valid_loss.append(validation_losses)


    # ===== Save Outputs to Timestamped Directory (once after all training) =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))  # path to this script
    save_dir = os.path.join(base_dir, "results", f"{timestamp}_unfolded")

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
