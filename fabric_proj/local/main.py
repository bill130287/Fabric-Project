import torchvision.datasets as dset 
import torchvision.transforms as transforms
import torch.utils.data as data
from tqdm import tqdm
import os
from phe import paillier
# from matplotlib import pyplot as plt

from module.FL_system import GenerateParticipant, SplitData, GlobalServer, Participant
from module.oracle import Oracle
from module.net import MLP, SqueezeNet
from module.util import DecryptDownloadParameters, preprocessDataset, WriteResult
import argparse

def CentralizeTraining(dataset, n=1):
    train_set, test_set = preprocessDataset(dataset=dataset)

    test_dataset = data.DataLoader(dataset = test_set, batch_size=64, shuffle=True)
    TrainData_dict, _ = SplitData(train_set, test_set, n)
    Participant_dict = GenerateParticipant(TrainData_dict, n)
    global_accuracy_list = []

    s = "P0"
    accu, _ = Participant_dict[s].LocalTesting(test_dataset)
    global_accuracy_list.append(accu)

    epochs = 50
    for _ in tqdm(range(epochs)): 
        Participant_dict[s].OnlyTraining()
        accu, _ = Participant_dict[s].LocalTesting(test_dataset)

        global_accuracy_list.append(accu)
        
    WriteResult(f"Centralize Training + {dataset}", global_accuracy_list, Participant_dict[s].validate_loss)


def FLTraining(dataset, n=15, in_clients=5, key_bits=256, _budget=4e-5, _isDP=False):
    main_dir = "./ExperimentFile/1" + f"/in_clients_{in_clients}" + f"/{dataset}"
    # if not os.path.exists(main_dir):
    #     os.makedirs(main_dir)    

    # Dataset
    train_set, test_set = preprocessDataset(dataset=dataset)

    init_params_dict_file = "/InitGlobalParamsPlainText.pickle"
    Server = GlobalServer(main_dir, init_params_dict_file, in_clients, isParamsFile=False)
    test_dataset = data.DataLoader(dataset = test_set, batch_size=64, shuffle=True)
    TrainData_dict, TestData_dict = SplitData(train_set, test_set, n)
    Participant_dict = GenerateParticipant(TrainData_dict, n)

    public_key1, private_key1 = paillier.generate_paillier_keypair(n_length=key_bits)
    public_key2, private_key2 = paillier.generate_paillier_keypair(n_length=key_bits)

    # # 檢測accuracy
    test_P = GenerateParticipant(TrainData_dict, 1) 

    global_accuracy_list = []
    # global_loss_history = []
    rounds = 50 #50 # round robin要做幾輪
    
    test_P["P0"].Download(Server)
    accu, _ = test_P["P0"].LocalTesting(test_dataset)
    global_accuracy_list.append(accu)

    for i in tqdm(range(rounds)):
        upload_encrypted_grad_dir = main_dir + "/round" + str(i + 1)
        for j in range(in_clients): # tqdm(range(in_clients)):
            s = 'P'
            s = s + str(j)
            
            encrypted_grad_file = f"/{s}_grads.json"
            noise_file = f"/{s}_noise.json"
            Participant_dict[s].LocalTraining(Server, public_key1, private_key1, public_key2, private_key2, \
                upload_encrypted_grad_dir, encrypted_grad_file, noise_file, epochs=1, budget=_budget, isGenerate=False, isDP=_isDP)
            Server.UpdateModel()
                
        test_P["P0"].Download(Server)
        accu, f1 = test_P["P0"].LocalTesting(test_dataset)
        print(f"accu: {accu}, f1_score: {f1}")
        global_accuracy_list.append(accu)
    
    WriteResult(f"Traditional FL + DP_{_isDP} + {dataset} + inclinets_{in_clients} + budget_{_budget}", global_accuracy_list, test_P["P0"].validate_loss)

def LocalTraining(dataset, n=15):
    # Dataset
    train_set, test_set = preprocessDataset(dataset=dataset)
    
    test_dataset = data.DataLoader(dataset = test_set, batch_size=64, shuffle=True)
    TrainData_dict, TestData_dict = SplitData(train_set, test_set, n)
    Participant_dict = GenerateParticipant(TrainData_dict, num=n)

    rounds = 50
    accu_list = []
    valloss_list = []

    # initial accu
    accu = 0
    val_loss = 0
    for j in range(n): # tqdm(range(in_clients)):
        s = 'P'
        s = s + str(j)
        # Participant_dict[s].OnlyTraining(epochs=1)
        tmp_accu, _ = Participant_dict[s].LocalTesting(test_dataset)
        # print(f"Participant {s}, accu: {tmp_accu}")
        accu += tmp_accu
        val_loss += Participant_dict[s].validate_loss[-1]
    accu /= n
    accu_list.append(accu)
    valloss_list.append(val_loss / n)

    for i in tqdm(range(rounds)):
        accu = 0
        val_loss = 0
        # print(f"round: {i}")
        for j in range(n): # tqdm(range(in_clients)):
            s = 'P'
            s = s + str(j)
            Participant_dict[s].OnlyTraining(epochs=1)
            tmp_accu, _ = Participant_dict[s].LocalTesting(test_dataset)
            # print(f"Participant {s}, accu: {tmp_accu}")
            accu += tmp_accu
            val_loss += Participant_dict[s].validate_loss[-1]
        accu /= n
        accu_list.append(accu)
        valloss_list.append(val_loss / n)
        print(f"accu: {accu}, val loss: {val_loss / n}")
    
    WriteResult(f"LocalTraining + {dataset} + clinets_{15}", accu_list, valloss_list)

def OursTrain(dataset, n=15, in_clients=5, key_bits=256, _budget=0.004):
    main_dir = "./ExperimentFile/2" + f"/in_clients_{in_clients}" + f"/{dataset}"
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)    

    # Dataset
    train_set, test_set = preprocessDataset(dataset=dataset)

    init_params_dict_file = "/InitGlobalModel.pickle"
    Server = GlobalServer(main_dir, init_params_dict_file, in_clients)
    test_dataset = data.DataLoader(dataset = test_set, batch_size=64, shuffle=True)
    TrainData_dict, _ = SplitData(train_set, test_set, n)
    Participant_dict = GenerateParticipant(TrainData_dict, num=n)

    public_key1, private_key1 = paillier.generate_paillier_keypair(n_length=key_bits)
    public_key2, private_key2 = paillier.generate_paillier_keypair(n_length=key_bits)

    with open(main_dir+"/key_info", 'w') as f:
        f.write(f"public1 N: {public_key1.n}\n")
        f.write(f"prime1 p: {private_key1.p}\n")
        f.write(f"prime1 q: {private_key1.q}\n")
        f.write(f"public2 N: {public_key2.n}\n")
        f.write(f"prime2 p: {private_key2.p}\n")
        f.write(f"prime2 q: {private_key2.q}\n")
    
    # # 檢測accuracy
    test_P = GenerateParticipant(TrainData_dict, 1) 

    # initial random encrypted params
    init_paramsfile = "/initparams.json"
    init_noisefile = "/initnoise.json"
    Server.GenEncryptedParams(public_key1, main_dir, init_paramsfile)
    Server.GenInitNoiseJson(public_key2, main_dir, init_noisefile)

    server_accu_list = []
    server_loss_history = []
    global_accuracy_list = []
    global_loss_history = []
    rounds = 50 # 要做幾輪
    
    test_P["P0"].Download(Server)
    accu, _ = test_P["P0"].LocalTesting(test_dataset)
    global_accuracy_list.append(accu)
    global_loss_history.append(test_P["P0"].validate_loss[-1])
    server_accu_list.append(accu)
    server_loss_history.append(test_P["P0"].validate_loss[-1])

    min_exponent = -30
    divide_exponent = -14
    exponent = min_exponent + divide_exponent
    oracle = Oracle(exponent, public_key2, private_key2, test_P["P0"].device, main_dir)
    param_file = "/download_params.json"
    noise_file = "/download_noises.json"

    import torch
    aggregate_noise = dict()
    # System start
    for r in range(rounds):  # while (round <= totalround):
        # Training
        upload_encrypted_grad_dir = main_dir + "/round" + str(r + 1)
        round_aggregate_noise = dict()
        for i in range(in_clients):
            s = 'P'
            s += str(i)
            encrypted_grad_file = f"/{s}_grads.json"
            round_noise_file = f"/{s}_noise.json"
            # print(test_P["P0"].local_network.state_dict()["fire8.expand_3x3.0.bias"])
            Participant_dict[s].local_network.load_state_dict(test_P["P0"].local_network.state_dict())
            # if i == 0:
            #     tmp_accu, _ = test_P["P0"].LocalTesting(test_dataset)
            #     print(f"testP accu: {tmp_accu}")
            #     tmp_accu, _ = Participant_dict[s].LocalTesting(test_dataset)
            #     print(f"{s} accu: {tmp_accu}")
            noise_dict = Participant_dict[s].LocalTraining(Server, public_key1, private_key1, public_key2, private_key2, \
                        upload_encrypted_grad_dir, encrypted_grad_file, round_noise_file, epochs=1, budget=_budget, isGenerate=True, isDP=True, isReturnNoise=True, isOurs=True)
            
            if i == 0:
                round_aggregate_noise = noise_dict
            else:
                for layer in round_aggregate_noise.keys():
                    round_aggregate_noise[layer] += noise_dict[layer]
            
            Server.UpdateModel()
        
        if (r == 0):
            for layer in round_aggregate_noise.keys():
                if round_aggregate_noise[layer].dtype == torch.int64:
                    aggregate_noise[layer] = (round_aggregate_noise[layer] / in_clients).to(torch.int64)
                else:
                    aggregate_noise[layer] = round_aggregate_noise[layer] / in_clients
        else:
            for layer in round_aggregate_noise.keys():
                if round_aggregate_noise[layer].dtype == torch.int64:
                    aggregate_noise[layer] +=  (round_aggregate_noise[layer] / in_clients).to(torch.int64)
                else:
                    aggregate_noise[layer] +=  round_aggregate_noise[layer] / in_clients
            
            # aggregate_noise[layer] /= in_clients
        
        # Central server testing
        test_P["P0"].Download(Server)
        accu, _ = test_P["P0"].LocalTesting(test_dataset)
        server_accu_list.append(accu)
        server_loss_history.append(test_P["P0"].validate_loss[-1])
        print(f"Central server accu with noise: {accu}")

        params_dict = test_P["P0"].local_network.state_dict()
        noiseAES_dict, cpabeAES_key, ivAES, cpabePK, cpabeMK = oracle.handlenoises_dict(aggregate_noise)
        nonoise_params = test_P["P0"].removenoise(params_dict, noiseAES_dict, cpabeAES_key, ivAES, cpabePK, cpabeMK)

        test_P["P0"].local_network.load_state_dict(nonoise_params)
        hyperledger_accu, _ = test_P["P0"].LocalTesting(test_dataset)
        global_accuracy_list.append(hyperledger_accu)
        global_loss_history.append(test_P["P0"].validate_loss[-1])
        print(f"hyperledger accu remove noise: {hyperledger_accu}")
    
    WriteResult(f"OursTrain(FL+DP) + {dataset} + inclinets_{in_clients} + budget_{_budget}", server_accu_list, server_loss_history)

def OursTesting(dataset, in_clients = 5):
    min_exponent = -30
    divide_exponent = -14
    exponent = min_exponent + divide_exponent

    main_dir = "./ExperimentFile/2" + f"/in_clients_{in_clients}" + f"/{dataset}" 
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)    

    # Dataset
    _, test_set = preprocessDataset(dataset)
    
    # key management
    with open(main_dir + "/key_info", 'r') as f:
        i = 0
        pk_list = list()
        p_list = list()
        q_list = list()

        for line in f.readlines():
            line = line.split(': ')
            # print(type(line[1].strip('\n')))
            if i%3 == 0:
                pk_list.append(int(line[1].strip('\n')))
            elif i%3 == 1:
                p_list.append(int(line[1].strip('\n')))
            else:
                q_list.append(int(line[1].strip('\n')))

            i += 1
    f.close()

    public_key1 = paillier.PaillierPublicKey(pk_list[0])
    private_key1 = paillier.PaillierPrivateKey(public_key1, p_list[0], q_list[0])
    public_key2 = paillier.PaillierPublicKey(pk_list[1])
    private_key2 = paillier.PaillierPrivateKey(public_key2, p_list[1], q_list[1])
    print(public_key1.n)
    print(public_key2.n)
    # Testing
    test_P = Participant("P1", None)
    oracle = Oracle(exponent, public_key2, private_key2, test_P.device, main_dir)
    test_dataset = data.DataLoader(dataset = test_set, batch_size=64, shuffle=True)
    param_file = "/download_params.json"
    noise_file = "/download_noises.json"
    
    import pickle
    path = main_dir + "/InitGlobalModel.pickle" # "/InitGlobalParamsPlainText.pickle" # f"/similaroursInitGlobalModel.pickle"
    with open(path, 'rb') as f:
        params_dict = pickle.load(f)

    accu_list = []
    test_P.local_network.load_state_dict(params_dict)
    accu, f1 = test_P.LocalTesting(test_dataset)
    accu_list.append(accu)
    print(f"init accu: {accu}, init f1: {f1}")

    rounds = 5
    for i in range(rounds):
        path = main_dir + f"/round{i + 1}" + param_file
        params_dict = DecryptDownloadParameters(path, exponent, public_key1, private_key1, test_P.device)
        # print(f"param conv10.bias: {params_dict['conv10.bias']}")
        noiseAES_dict, cpabeAES_key, ivAES, cpabePK, cpabeMK = oracle.handlenoises(f"/round{i + 1}" + noise_file)
        nonoise_params = test_P.removenoise(params_dict, noiseAES_dict, cpabeAES_key, ivAES, cpabePK, cpabeMK)
        
        test_P.local_network.load_state_dict(nonoise_params)
        accu, f1 = test_P.LocalTesting(test_dataset)
        accu_list.append(accu)
        print(f"round: {i + 1}, accu: {accu}, f1 score:{f1}")


        # print(f"accu: {accu}")

    # for l in nonoise_params.keys():
    #     if (not torch.equal(prev[l], nonoise_params[l])):
    #         print(f"layer: {l}")
    WriteResult(f"Fabric FL Ours Testing  + {dataset} + inclinets_{in_clients}", accu_list, test_P.validate_loss)

def _get_parser():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--dataset', type=str, default="MNIST", metavar='D', help='dataset: MNIST, CIFAR10, CIFAR100')
    parser.add_argument('--n', type=int, default=15, metavar='C', help='number of clients')
    parser.add_argument('--inclients', type=int, default=5, metavar='IC', help='nuumber of clients training')
    parser.add_argument('--keybits', type=int, default=256, metavar='K', help='Pallier key bits')
    parser.add_argument('--budget', type=float, default=4e-5, help='privacy budget')
    parser.add_argument('--mode', type=int, default=1, help="training mode: 1:federated training(FL), 2: FL with differential privacy, \
                        3:centralize training, 4:local training, 5:ours")
    parser.add_argument('--test', action='store_true', default=False, help='testing in ours')

    return parser

if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args() 
    
    if not args.test:
        if args.mode == 1:
            FLTraining(dataset=args.dataset, n=args.n, in_clients=args.inclients, key_bits=args.keybits, _isDP=False) # FL 
        elif args.mode == 2:
            FLTraining(dataset=args.dataset, n=args.n, in_clients=args.inclients, key_bits=args.keybits, _budget=args.budget, _isDP=True) # FL + DP
        elif args.mode == 3:
            CentralizeTraining(dataset=args.dataset)
        elif args.mode == 4:
            LocalTraining(dataset=args.dataset, n=args.n)
        elif args.mode == 5:
            OursTrain(dataset=args.dataset, n=args.n, in_clients=args.inclients, key_bits=args.keybits, _budget=args.budget)
        else:
            raise TypeError(f"mode {args.mode} does not exist")
    else:
        OursTesting(dataset=args.dataset, in_clients=args.inclients)