import os
import argparse
import random
import copy
import csv
import torch
from pathlib import Path

import setupGC
from training import *
import matplotlib.ticker as ticker


def process_selftrain(clients, server, local_epoch):
    print("Self-training ...")
    df = pd.DataFrame()
    allAccs = run_selftrain_GC(clients, server, local_epoch)
    for k, v in allAccs.items():
        df.loc[k, [f'train_acc', f'val_acc', f'test_acc']] = v
    print(df)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_selftrain_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_selftrain_GC{suffix}.csv')
    df.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_fedavg(clients, server):
    print("\nDone setting up FedAvg devices.")

    print("Running FedAvg ...")
    frame = run_fedavg(clients, server, args.num_rounds, args.local_epoch, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedavg_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedavg_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_fedprox(clients, server, mu):
    print("\nDone setting up FedProx devices.")

    print("Running FedProx ...")
    frame = run_fedprox(clients, server, args.num_rounds, args.local_epoch, mu, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedprox_mu{mu}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedprox_mu{mu}_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_gcfl(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcfl_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcfl_GC{suffix}.csv')

    frame = run_gcfl(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_gcflplus(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL plus ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcflplus_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcflplus_GC{suffix}.csv')

    frame = run_gcflplus(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2, args.seq_length, args.standardize)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_gcflplusdWs(clients, server):
    print("\nDone setting up CFL devices.")
    print("Running GCFL plus dWs ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcflplusDWs_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcflplusDWs_GC{suffix}.csv')

    frame = run_gcflplus_dWs(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2, args.seq_length, args.standardize)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
    parser.add_argument('--num_repeat', type=int, default=5,
                        help='number of repeating rounds to simulate;')
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='number of rounds to simulate;')
        
    parser.add_argument('--comm_threshold', type=int, default=10,
                        help='the communication round threshold to cluster;')
    parser.add_argument('--dist_threshold',type=float,default=1.0,
                        help='the distance threshold to cluster')
    
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of GINconv layers')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=123)

    parser.add_argument('--datapath', type=str, default='./data',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs_csv',
                        help='The base path for outputting.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='mix')

    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--standardize', help='whether to standardize the distance matrix',
                        type=bool, default=False)
    parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=10)
    parser.add_argument('--epsilon1', help='the threshold epsilon1 for GCFL',
                        type=float, default=0.01)
    parser.add_argument('--epsilon2', help='the threshold epsilon2 for GCFL',
                        type=float, default=0.1)

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    seed_dataSplit = 123

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    EPS_1 = args.epsilon1
    EPS_2 = args.epsilon2

    # TODO: change the data input path and output path
    outbase = os.path.join(args.outbase, f'seqLen{args.seq_length}')

    if args.overlap and args.standardize:
        outpath = os.path.join(outbase, f"standardizedDTW/multiDS-overlap")
    elif args.overlap:
        outpath = os.path.join(outbase, f"multiDS-overlap")
    elif args.standardize:
        outpath = os.path.join(outbase, f"standardizedDTW/multiDS-nonOverlap")
    else:
        outpath = os.path.join(outbase, f"multiDS-nonOverlap")
    outpath = os.path.join(outpath, args.data_group, f'eps_{EPS_1}_{EPS_2}')
    Path(outpath).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outpath}")

    # preparing data
    if not args.convert_x:
        """ using original features """
        suffix = ""
        print("Preparing data (original features) ...")
    else:
        """ using node degree features """
        suffix = "_degrs"
        print("Preparing data (one-hot degree features) ...")

    if args.repeat is not None:
        Path(os.path.join(outpath, 'repeats')).mkdir(parents=True, exist_ok=True)

    splitedData, df_stats = setupGC.prepareData_multiDS(args.datapath, args.data_group, args.batch_size, convert_x=args.convert_x, seed=seed_dataSplit)
    print("Done")

    # save statistics of data on clients
    if args.repeat is None:
        outf = os.path.join(outpath, f'stats_trainData{suffix}.csv')
    else:
        outf = os.path.join(outpath, "repeats", f'{args.repeat}_stats_trainData{suffix}.csv')
    df_stats.to_csv(outf)
    print(f"Wrote to {outf}")

    init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
    print("\nDone setting up devices.")

    # process_selftrain(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), local_epoch=100)
    # process_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    _ , mean_accs_fedavg = run_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), COMMUNICATION_ROUNDS = args.num_rounds , local_epoch = args.local_epoch)
    _ , mean_accs_ghcfl = run_ghcfl(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), comm_threshold =args.comm_threshold, dist_threshold = args.dist_threshold, COMMUNICATION_ROUNDS = args.num_rounds, local_epoch =args.local_epoch)
    _ , mean_accs_fedprox = run_fedprox(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), mu=0.01, COMMUNICATION_ROUNDS = args.num_rounds, local_epoch =args.local_epoch)
    # process_fedprox(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), mu=0.01)
    # process_gcfl(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    # process_gcflplus(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    # process_gcflplusdWs(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))

    fig, ax = plt.subplots()
    df = pd.DataFrame()
    df_test_dist = pd.DataFrame()
    ax.plot(mean_accs_ghcfl, label='GHCFL')
    ax.plot(mean_accs_fedavg, label='FedAvg')
    ax.plot(mean_accs_fedprox, label='FedProx')
    print("mean_accs_fedavg",mean_accs_fedavg)
    # ax.plot(mean_accs_fedprox,label = 'FedProx')
    # ax.plot(mean_accs_gcfl,label = "GCFL")
    plt.axvline(args.comm_threshold-1,color='r', linestyle='--', label='n')
    # plt.ylim((0.2, 1.4))
    plt.xticks(np.arange(len(mean_accs_ghcfl)), np.arange(1, len(mean_accs_ghcfl)+1))
    plt.xticks(np.arange(len(mean_accs_fedavg)), np.arange(1, len(mean_accs_fedavg)+1))
    plt.xticks(np.arange(len(mean_accs_fedprox)), np.arange(1, len(mean_accs_fedprox)+1))
    plt.xticks(rotation=45)   # 设置横坐标显示的角度，角度是逆时针，自己看
    tick_spacing = 3 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend() #自动检测要在图例中显示的元素，并且显示
    
    max_acc_ghcfl = np.max(mean_accs_ghcfl)
    max_acc_fedavg = np.max(mean_accs_fedavg)
    max_acc_fedprox = np.max(mean_accs_fedprox)

    figure_save_path = "file_figs"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
    r = args.num_rounds
    n = args.comm_threshold
    name = args.data_group
    plt.savefig(os.path.join(figure_save_path , f'{r}r_{n}n_{name}_{args.local_epoch}e_{args.dist_threshold}d.png'))
    # plt.show() #图形可视化
        
    mean_acc_fedavg = np.mean(mean_accs_fedavg)
    mean_acc_fedprox = np.mean(mean_accs_fedprox)
    mean_acc_ghcfl = np.mean(mean_accs_ghcfl)

    improve_ratio_ghcfl = mean_acc_ghcfl/mean_acc_fedavg
    improve_ratio_fedprox = mean_acc_fedprox/mean_acc_fedavg

    df.loc[f'{args.data_group}', 'improve_ratio_to_avg'] = improve_ratio_ghcfl
    df.loc[f'{args.data_group}', 'improve_ratio_to_prox'] = improve_ratio_fedprox
    df.loc[f'{args.data_group}', 'max_acc_ghcfl'] = max_acc_ghcfl
    df.loc[f'{args.data_group}', 'max_acc_fedavg'] = max_acc_fedavg
    df.loc[f'{args.data_group}', 'max_acc_fedprox'] = max_acc_fedprox
    
    outbase = './outputs_csv'
    # 如果不存在目录figure_save_path，则创建
    outbase = os.path.join(args.outbase, f'localEpoch{args.local_epoch}')
    outpath = os.path.join(outbase, f"distThreshold{args.dist_threshold}")
    outfile = os.path.join(outpath, f'{args.num_rounds}r_{args.comm_threshold}n_{args.data_group}.csv')
    if not os.path.exists(outpath):
            os.makedirs(outpath)
    df.to_csv(outfile)

    filename = f'{args.data_group}_dist_test.csv'

    # 如果文件不存在，则创建文件
    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            f.write('')  # 写入空字符串
    if os.path.getsize(filename) == 0:
        length = 0
    else :
        length = 1
    df = pd.DataFrame()
    print(length)
    if length != 0:
        df = pd.read_csv(filename, header=0)
        # df = df.drop(df.columns[0], axis=1)
        df = df.assign(**{f'{args.data_group}-{args.dist_threshold}d': mean_accs_ghcfl})
        df.to_csv(filename,index=False)
    else:
        df = df.assign(**{f'{args.data_group}-{args.dist_threshold}d': mean_accs_ghcfl})
        df.to_csv(filename,index=False)