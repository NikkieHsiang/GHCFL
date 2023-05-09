import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

mean_accs_fedavg = []
mean_accs_ghcfl = []
mean_accs_fedprox = []
clients_over_ghcfl = []
dw_ghcfl = []
ratios = {}
last_acc_fedavg = []
last_acc_fedprox = []
last_acc_gcfl = []
last_acc_ghcfl = []

def run_selftrain_GC(clients, server, local_epoch):
    # all clients are initialized with the same weights
    for client in clients:
        client.download_from_server(server)

    allAccs = {}
    for client in clients:
        client.local_train(local_epoch)

        loss, acc = client.evaluate()
        allAccs[client.name] = [client.train_stats['trainingAccs'][-1], client.train_stats['valAccs'][-1], acc]
        print("  > {} done.".format(client.name))

    return allAccs

def run_ghcfl(clients, server, comm_threshold, dist_threshold, COMMUNICATION_ROUNDS, local_epoch, frac=1.0):
    print("running hierarchical clustering...")
    
    '''initialize w_0'''
    for client in clients:
        client.download_from_server(server)#initialize all the clients 
    '''for round 1 - n, do fedAvg selecting a partial clients'''
    for c_round in range(1,comm_threshold):#1-n round communication, do FEDERATED_LEARNING
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = server.randomSample_clients(clients, frac)
            #21:28
        for client in selected_clients:
            client.local_train(local_epoch) #client_update
        
        server.aggregate_weights(selected_clients) #W(t+1) = sum(n{k}/n * W(t))
        
        #pass the aggregated weights from server to selected clients:
        for client in selected_clients:
            client.download_from_server(server)#到了这一步，客户端最后存储了server进行Avg操作后的joint global model
            # cache the aggregated weights for next round
            # client.cache_weights()#这里已经把joint global model的w存到了w_old
        
        accs = []
        client_dWs = []         
        for client in clients:
            loss, acc = client.evaluate()
            accs.append(acc)
            dW = []
            for k in client.W.keys():
                dW.append(client.dW[k])
            client_dWs.append(dW)
        mean_accs_ghcfl.append(np.mean(accs))#每一个round计算一次所有clients的mean_acc
        # dw_ghcfl.append(np.mean(client_dWs))
        clients_over_ghcfl.append(len([acc for acc  in accs if acc > 0.75]))
    # print("\nmean_accs:",mean_accs_ghcfl)
    
    
    '''for each of all clients, do CLIENT_UPDATE to get delta W'''
    for client in clients:
        client.compute_weight_update(local_epoch)
    '''server cluster the clients by hierarchical_clustering_algorithm'''
    client_clusters = server.hierarchical_clustering_clients(clients, dist_threshold)
    
    
    '''for the rest of the communication rounds, update by clusters'''
    # for client in clients:
    #     client.download_from_server(server)#initialize all the clients 
    for c_round in range(comm_threshold,COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        selected_clients = server.randomSample_clients(clients, frac)
        
        for client in selected_clients:
            client.compute_weight_update(local_epoch) #client_update
            
        server.aggregate_clusterwise(client_clusters)#aggregate by clusters
        #pass the aggregated weights from server to selected clients:
        # for client in selected_clients:
            # client.download_from_server(server)#到了这一步，客户端最后存储了server进行Avg操作后的joint global model
            # cache the aggregated weights for next round
            # client.cache_weights()#这里已经把joint global model的w存到了w_old
            
        accs = [] 
        client_dWs = []    
        for client in clients:
            loss, acc = client.evaluate()
            accs.append(acc)
            dW = []
            for k in client.W.keys():
                dW.append(client.dW[k])
            client_dWs.append(dW)
        mean_accs_ghcfl.append(np.mean(accs))#每一个round计算一次所有clients的mean_acc
        # dw_ghcfl.append(np.mean(client_dWs))
        # if c_round == COMMUNICATION_ROUNDS:
        #     print("final accs:",accs) 
        #     last_acc_ghcfl = accs
            
        clients_over_ghcfl.append(len([acc for acc  in accs if acc > 0.75]))
    # last_acc_ghcfl = accs 
    
    ratios["HCFL"] = clients_over_ghcfl
    print("\nmean_accs:",mean_accs_ghcfl)
   
   
    '''test and output results to file'''
    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    print(frame)
    return frame,mean_accs_ghcfl

def run_fedavg(clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            # only get weights of graphconv layers
            client.local_train(local_epoch)

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(server)
        accs = []   
        for client in clients:
            loss, acc = client.evaluate()
            accs.append(acc)
        mean_accs_fedavg.append(np.mean(accs))
        if c_round == COMMUNICATION_ROUNDS:
            print("final accs:",accs) 
            last_acc_fedavg = accs
            
    clients_over_avg = len([acc for acc  in accs if acc > 0.75])
    ratios["FedAvg"] = clients_over_avg
    
    # plt.plot(mean_accs)
     
    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame,mean_accs_fedavg


# def run_fedavg(clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0):
#     for client in clients:
#         client.download_from_server(server)

#     if samp is None:
#         sampling_fn = server.randomSample_clients
#         frac = 1.0
#     mean_accs = []
#     for c_round in range(1, COMMUNICATION_ROUNDS + 1):
#         if (c_round) % 50 == 0:
#             print(f"  > round {c_round}")

#         if c_round == 1:
#             selected_clients = clients
#         else:
#             selected_clients = sampling_fn(clients, frac)

#         for client in selected_clients:
#             # only get weights of graphconv layers
#             client.local_train(local_epoch)

#         server.aggregate_weights(selected_clients)
#         for client in selected_clients:
#             client.download_from_server(server)
#         accs = []   
#         for client in clients:
#             loss, acc = client.evaluate()
#             accs.append(acc)
#         mean_accs.append(np.mean(accs))

#     frame = pd.DataFrame()
#     for client in clients:
#         loss, acc = client.evaluate()
#         frame.loc[client.name, 'test_acc'] = acc

#     def highlight_max(s):
#         is_max = s == s.max()
#         return ['background-color: yellow' if v else '' for v in is_max]

#     fs = frame.style.apply(highlight_max).data
#     print(fs)
#     return frame

def run_fedprox(clients, server, COMMUNICATION_ROUNDS, local_epoch, mu, samp=None, frac=1.0):
    for client in clients:
        client.download_from_server(server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    if samp == 'random':
        sampling_fn = server.randomSample_clients

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            client.local_train_prox(local_epoch, mu)

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(server)

            # cache the aggregated weights for next round
            client.cache_weights()
        accs = []   
        for client in clients:
            loss, acc = client.evaluate()
            accs.append(acc)
        mean_accs_fedprox.append(np.mean(accs))
        if c_round == COMMUNICATION_ROUNDS:
            print("final accs:",accs) 
            last_acc_fedprox = accs
    clients_over_prox = len([acc for acc  in accs if acc > 0.85])
    ratios["FedProx"] = clients_over_prox
    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame,mean_accs_fedprox


# def run_fedprox(clients, server, COMMUNICATION_ROUNDS, local_epoch, mu, samp=None, frac=1.0):
#     for client in clients:
#         client.download_from_server(server)

#     if samp is None:
#         sampling_fn = server.randomSample_clients
#         frac = 1.0
#     if samp == 'random':
#         sampling_fn = server.randomSample_clients

#     for c_round in range(1, COMMUNICATION_ROUNDS + 1):
#         if (c_round) % 50 == 0:
#             print(f"  > round {c_round}")

#         if c_round == 1:
#             selected_clients = clients
#         else:
#             selected_clients = sampling_fn(clients, frac)

#         for client in selected_clients:
#             client.local_train_prox(local_epoch, mu)

#         server.aggregate_weights(selected_clients)
#         for client in selected_clients:
#             client.download_from_server(server)

#             # cache the aggregated weights for next round
#             client.cache_weights()

#     frame = pd.DataFrame()
#     for client in clients:
#         loss, acc = client.evaluate()
#         frame.loc[client.name, 'test_acc'] = acc

#     def highlight_max(s):
#         is_max = s == s.max()
#         return ['background-color: yellow' if v else '' for v in is_max]

#     fs = frame.style.apply(highlight_max).data
#     print(fs)
#     return frame


def run_gcfl(clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1, EPS_2):
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
 
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(server)

        participating_clients = server.randomSample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.compute_weight_update(local_epoch)
            client.reset()

        similarities = server.compute_pairwise_similarities(clients)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20:
                server.cache_model(idc, clients[idc[0]].W, acc_clients)
                print("\nbi-partition occurs\n")
                c1, c2 = server.min_cut(similarities[idc][:, idc], idc)
                cluster_indices_new += [c1, c2]
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]  # initial: [[0, 1, ...]]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate()[1] for client in clients] #每个client的acc_num/graphs_num

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(results, columns=["FL Model"] + ["Model {}".format(i)
                                                          for i in range(results.shape[1] - 1)],
                         index=["{}".format(clients[i].name) for i in range(results.shape[0])])
    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ['test_acc']
    print(frame)

    return frame


def run_gcflplus(clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1, EPS_2, seq_length, standardize):
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads = {c.id:[] for c in clients}#初始化梯度字典，key：client.id, value: client.convGradsNorm
    for client in clients:
        client.download_from_server(server)

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(server)

        participating_clients = server.randomSample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.compute_weight_update(local_epoch)
            client.reset()

            seqs_grads[client.id].append(client.convGradsNorm)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20 \
                    and all(len(value) >= seq_length for value in seqs_grads.values()):

                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances)-dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate()[1] for client in clients]

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(results, columns=["FL Model"] + ["Model {}".format(i)
                                                          for i in range(results.shape[1] - 1)],
                         index=["{}".format(clients[i].name) for i in range(results.shape[0])])

    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ['test_acc']
    print(frame)

    return frame


def run_gcflplus_dWs(clients, server, COMMUNICATION_ROUNDS, local_epoch, EPS_1, EPS_2, seq_length, standardize):
    cluster_indices = [np.arange(len(clients)).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    seqs_grads = {c.id:[] for c in clients}
    for client in clients:
        client.download_from_server(server)

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        if c_round == 1:
            for client in clients:
                client.download_from_server(server)

        participating_clients = server.randomSample_clients(clients, frac=1.0)

        for client in participating_clients:
            client.compute_weight_update(local_epoch)
            client.reset()

            seqs_grads[client.id].append(client.convDWsNorm)

        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 20 \
                    and all(len(value) >= seq_length for value in seqs_grads.values()):

                server.cache_model(idc, clients[idc[0]].W, acc_clients)

                tmp = [seqs_grads[id][-seq_length:] for id in idc]
                dtw_distances = server.compute_pairwise_distances(tmp, standardize)
                c1, c2 = server.min_cut(np.max(dtw_distances)-dtw_distances, idc)
                cluster_indices_new += [c1, c2]

                seqs_grads = {c.id: [] for c in clients}
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate()[1] for client in clients]

    for idc in cluster_indices:
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(results, columns=["FL Model"] + ["Model {}".format(i)
                                                          for i in range(results.shape[1] - 1)],
                         index=["{}".format(clients[i].name) for i in range(results.shape[0])])
    frame = pd.DataFrame(frame.max(axis=1))
    frame.columns = ['test_acc']
    print(frame)

    return frame
