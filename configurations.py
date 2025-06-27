import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--m', type=int, default=100,
                        help="size of m")
    parser.add_argument('--n', type=int, default=500,
                        help="size of n")
    parser.add_argument('--alpha_max', type=float, default=0.1,
                        help="max value of alpha (reduced for high GHN_iter_num)")
    parser.add_argument('--tau_max', type=float, default=0.99,
                        help="max value of tau (reduced for high GHN_iter_num)")
    parser.add_argument('--rho_max', type=float, default=0.99,
                        help="max value of rho (reduced for high GHN_iter_num)")
    parser.add_argument('--eta_max', type=float, default=0.99,
                        help="max value of eta (reduced for high GHN_iter_num)")

    # Hyperparameter initialization options
    parser.add_argument('--init_alpha_frac', type=float, default=0.2,
                        help="initial alpha as fraction of alpha_max (0.0-1.0)")
    parser.add_argument('--init_tau_frac', type=float, default=0.15,
                        help="initial tau as fraction of tau_max (0.0-1.0)")
    parser.add_argument('--init_rho_frac', type=float, default=0.25,
                        help="initial rho as fraction of rho_max (0.0-1.0)")
    parser.add_argument('--init_eta_frac', type=float, default=0.1,
                        help="initial eta as fraction of eta_max (0.0-1.0)")
    parser.add_argument('--max_penalty_threshold', type=float, default=0.8,
                        help="threshold for applying hyperparameter regularization")
    parser.add_argument('--penalty_reduction_factor', type=float, default=0.95,
                        help="factor to reduce hyperparameters when penalty is applied")

    parser.add_argument('--exp_name', type=str, default='exp for 5 agents',
                        help="the name of the current experiment")
    parser.add_argument('--eval', action='store_true',
                        help="whether to perform inference of training")
    parser.add_argument('--method', type=str, default='u-dadmm',
                        help='choose which method you want to use dadmm or u-dadmm')
    parser.add_argument('--seq_num', type=int, default=0,
                        help="Use to save the best model in each sequential training,"
                             "\nif max_iter_seg == max_iter there is no sequential training")

    # data arguments
    parser.add_argument('--data', type=str, default='simulated',
                        choices=['mnist', 'simulated'],
                        help="dataset to use (mnist for linear regression or simulated for LASSO)")
    parser.add_argument('--norm_mean', type=float, default=0.5,
                        help="normalize the data to norm_mean")
    parser.add_argument('--norm_std', type=float, default=0.5,
                        help="normalize the data to norm_std")
    parser.add_argument('--train_size', type=int, default=200,
                        help="train dataset size")
    parser.add_argument('--snr', type=int, default=4,
                        help="Simulated dataset with SNR values {-2, 0, 2, 4}")
    parser.add_argument('--test_size', type=int, default=32,
                        help="test dataset size")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="trainset batch size")

    # Graph arguments
    parser.add_argument('--P', type=int, default=5, help="number of agents")
    parser.add_argument('--graph_prob', type=float, default=0.5,
                        help="Probability of connection between agents")
    parser.add_argument('--graph_type', type=str, default='erods_renyi',
                        help="Choose graph type e.g.,'erods_renyi', 'geometric', etc.")
    # DADMM arguments
    parser.add_argument('--case', type=str, default='dlasso',
                        choices=['dlasso', 'dlr'],
                        help="case to use (distributed LASSO or distributed linear regression)")
    parser.add_argument('--model', type=str, default='same',
                        choices=['diff', 'same'],
                        help="model to use (different or same hyperparameters)")
    parser.add_argument('--rho', type=float, default=0.2603,
                        help="rho hyperparameter (0.175 for dlasso and 2.4231e-05 for dlr)")
    parser.add_argument('--alpha', type=float, default=0.3013,
                        help="alpha hyperparameter (0.125 for dlasso and 4.3877e-04 for dlr)")
    parser.add_argument('--eta', type=float, default=0.0867,
                        help="eta hyperparameter (0.0867 for dlasso and 1.1221e-07 for dlr)")
    parser.add_argument('--gamma', type=float, default=1.1797e-07,
                        help="gamma hyperparameter (use only for dlr)")
    parser.add_argument('--beta', type=float, default=1.2260e-03,
                        help="beta hyperparameter (use only for dlr)")
    parser.add_argument('--delta', type=float, default=1.2665e-04,
                        help="delta hyperparameter (use only for dlr)")
    parser.add_argument('--tau', type=float, default=0.1142,
                        help="tau hyperparameter (use only for dlasso)")
    parser.add_argument('--sequential', type=bool, default=False,
                        help="Train with or without sequential training")
    parser.add_argument('--max_iter_seg', type=int, default=2,
                        help="number of train segments (No. of DADMM iterations)")
    parser.add_argument('--max_iter', type=int, default=25,
                        help="number of DADMM iterations")

    parser.add_argument('--num_epochs', type=int, default=10,
                        help="number of epochs")

    # learning arguments
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help="optimizer to use (sgd or adam)")
    parser.add_argument('--lr', type=float, default=1e-04,
                        help="optimizer learning rate")
    parser.add_argument('--momentum', type=float, default=0.5 * 1e-05,
                        help="momentum")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="weight decay")
    parser.add_argument('--lr_scheduler', action='store_true',
                        help="reduce the learning rat when val_acc has stopped improving (increasing)")
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda:0', 'cuda:1', 'cpu'],
                        help="device to use (gpu or cpu)")
    parser.add_argument('--valid', type=bool, default=True,
                        help='choose validation case')
    parser.add_argument('--seed', type=float, default=42,
                        help="manual seed for reproducibility")

    #gnn_arguments

    parser.add_argument('--GHyp_hidden', type=float, default=100,
                        help="Graph HyperNetwork hidden layers")
    parser.add_argument('--DADMM_mode', type=str, default='diff',
                        choices=['same','diff'],
                        help="GNN_Hyp DLASSO type")
    parser.add_argument('--hyp_mode', type=str, default='unfolded',
                        choices=['GHyp','unfolded'],
                        help="GNN_Hyp DLASSO type")
    parser.add_argument('--GHN_iter_num', type=int, default=15,
                        help="Number of DADMM iterations for the gnn hyp model (use <=20 for stability)")
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save models and logs')

    args = parser.parse_args()
    return args
