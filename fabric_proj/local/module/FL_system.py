import struct
import numpy as np
import math
from multiprocessing import Pool
from functools import partial
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from sklearn.metrics import f1_score
import pickle
from copy import deepcopy

from module.multithread_crypto import Encrypt_Multithread, Decrypt_Multithread
from module.util import POOL_CORES, WriteGradientsJson, WriteParametersJson
from module.net import MLP
# POOL_CORES = 16
# parameters exponent 為 MIN_EXPONENT
# gradiets 和 noise 為 MIN_GRADS_EXPONENT
# 因為 gradients 需要除以constant 所以expoent會減掉14
MIN_EXPONENT = -44
MIN_GRADS_EXPONENT = -30

class GlobalServer():
    def __init__(self, dir, file, aggregate_num, isParamsFile=True, net=MLP()):# MLP() # SqueezeNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_network_dict = self.Initial_globalnetwork_dict(net, dir, file, isParamsFile) #{"key is layer": value is numpy(from tensor to numpy)}
        self.gradient_buffer = dict()
        self.Inital_gradient_buffer()

        #self.lr = 0.1
        self.loss_history = []
        self.cnt = 0
        self.aggregate_num = aggregate_num

    def Initial_globalnetwork_dict(self, net, dir, file, isParamsFile):
        global_network = net.to(self.device)
        net_dict = global_network.state_dict()

        # for key in net_dict.keys():
        #     net_dict[key] = net_dict[key].numpy()
        
        if isParamsFile:
            path = dir + file
            with open(path, 'wb') as f:
                pickle.dump(net_dict, f)
        
        return net_dict

    def Inital_gradient_buffer(self):
        for layer in self.global_network_dict.keys():
            self.gradient_buffer[layer] = torch.zeros_like(self.global_network_dict[layer])

    def UpdateModel(self):
        if self.cnt != self.aggregate_num:
            return
        # print("Update")     
        # start = time.time()
        for layer in self.global_network_dict.keys():
            # print(f"layer: {layer}, type: {(self.gradient_buffer[layer] / self.aggregate_num).dtype}")
            # print(f"network type: {self.global_network_dict[layer].dtype}")
            if self.global_network_dict[layer].dtype == torch.int64:
                # print(f"track: {self.gradient_buffer[layer]}")
                self.global_network_dict[layer] += (self.gradient_buffer[layer] / self.aggregate_num).to(torch.int64)
            else:   
                self.global_network_dict[layer] += (self.gradient_buffer[layer] / self.aggregate_num)
        
        self.Inital_gradient_buffer()
        self.cnt = 0
        # end = time.time()
        # print(f"Update time: {end - start} sec")

    def AddGradient(self):
        state_dict_tmp = self.global_network_dict
        for key in self.gradient_buffer.keys():
            #print(f"Add layer:{key}")
            #print(key)
            state_dict_tmp[key] += self.gradient_buffer[key]
          
        self.global_network_dict = state_dict_tmp
        
        # WriteGradientsJson(self.gradient_buffer, MIN_EXPONENT, path="test/upload_testJson.json")
        # self.gradient_buffer.clear()

    def Append_loss(self, loss_sum):
        self.loss_history.append(loss_sum)

    def EncryptParameters(self, public_key):
        params_dict = self.global_network_dict
        
        pool = Pool(POOL_CORES)
        partial_func = partial(Encrypt_Multithread, public_key=public_key)
        for layer in params_dict.keys():
            shape = params_dict[layer].shape
            params_array = pool.map(partial_func, params_dict[layer].reshape(-1))
            params_concatenate = np.concatenate(params_array)
            params_dict[layer] = params_concatenate.reshape(shape)

        ''' # Original
        for key in params_dict.keys():
          shape = params_dict[key].shape
          params_array = np.array([public_key.encrypt(float(x)) for x in params_dict[key].reshape(-1)])
          params_dict[key] = params_array.reshape(shape)
        '''
        self.global_network_dict = params_dict
    
    def GenEncryptedParams(self, public_key, dir, file):
        params_dict = deepcopy(self.global_network_dict)
        shape_dict = dict()

        pool = Pool(POOL_CORES)
        partial_func = partial(Encrypt_Multithread, public_key=public_key)
        print('=' * 10 + "Encrypting Global Parameters" + '=' * 10)
        for layer in tqdm(params_dict.keys()):
            shape = params_dict[layer].cpu().numpy().shape
            shape_dict[layer] = shape
            params_array = pool.map(partial_func, params_dict[layer].cpu().numpy().reshape(-1))
            params_concatenate = np.concatenate(params_array)
            params_dict[layer] = params_concatenate.reshape(-1)
        
        print('=' * 10 + "Decrease Exponent" + '=' * 10)
        for layer in tqdm(params_dict.keys()):
            for i, cipher_params in enumerate(params_dict[layer]):
                params_dict[layer][i] = cipher_params.decrease_exponent_to(MIN_EXPONENT)

        WriteParametersJson(params_dict, shape_dict, MIN_EXPONENT, dir, file)
        
    def GenInitNoiseJson(self, public_key, dir, file):
        noise_dict = dict()
        shape_dict = dict()

        pool = Pool(POOL_CORES)
        partial_func = partial(Encrypt_Multithread, public_key=public_key)
        print('=' * 10 + "Encrypting Noise" + '=' * 10)
        for layer in tqdm(self.global_network_dict.keys()):
            shape = self.global_network_dict[layer].cpu().numpy().shape
            shape_dict[layer] = shape
            noise = torch.zeros_like(self.global_network_dict[layer])
            noise_array = pool.map(partial_func, noise.cpu().numpy().reshape(-1))
            noise_concatenate = np.concatenate(noise_array)
            noise_dict[layer] = noise_concatenate.reshape(-1)
            # partial_func2 = partial(Decrypt_Multithread, private_key=private_key)
            # decrypt_array = pool.map(partial_func2, noise_dict[layer])
            # print(decrypt_array)
        
        print('=' * 10 + "Decrease Exponent" + '=' * 10)
        for layer in tqdm(noise_dict.keys()):
            for i, cipher_params in enumerate(noise_dict[layer]):
                noise_dict[layer][i] = cipher_params.decrease_exponent_to(MIN_GRADS_EXPONENT)
        
        WriteParametersJson(noise_dict, shape_dict, MIN_GRADS_EXPONENT, dir, file)


class Participant():
    def __init__(self, name, training_data):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.local_network = MLP() #MLP()#LightVGG() #LeNet() #network()
        # from torchsummary import summary
        # summary(SqueezeNet().cuda(), (3, 32, 32))

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.local_network = self.local_network.cuda()

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = 0.001 #0.01 # 0.001
        self.optimizer = torch.optim.Adam(params=self.local_network.parameters(), lr = self.lr)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 60, 80], gamma=0.6)

        self.training_data = training_data
        self.loss_history = []
        self.validate_loss = []

    # ================================================= Download and upload all gradients =================================================== #
    def Download(self, GlobalServer):
        download_params = dict()
        # start = time.time()
        for layer in GlobalServer.global_network_dict.keys():
            download_params[layer] = GlobalServer.global_network_dict[layer]
        # end = time.time()
        self.local_network.load_state_dict(download_params)
        
        # start = time.time()
        # self.local_network.load_state_dict(GlobalServer.global_network_dict)
        

        # print(f"Download time: {end - start}")
        
    def DownloadDecrypt(self, GlobalServer, private_key):
        params_dict = GlobalServer.global_network_dict
        decrypt_params = dict()
        pool = Pool(POOL_CORES)
        partial_func = partial(Decrypt_Multithread, private_key=private_key)
        for key in params_dict.keys():
            shape = params_dict[key].shape
            decrypted_array = pool.map(partial_func, params_dict[key].reshape(-1))
            decrypt_params[key] = torch.Tensor(np.concatenate(decrypted_array).reshape(shape))

        pool.close()
        pool.join()
        self.local_network.load_state_dict(decrypt_params)

    def Upload(self, gradient_dict, GlobalServer):
        # start = time.time()
        for layer in gradient_dict.keys():
            # print(type(GlobalServer.gradient_buffer[layer]))
            # print(type(gradient_dict[layer]))
            GlobalServer.gradient_buffer[layer] += gradient_dict[layer]

        GlobalServer.cnt += 1
        # end = time.time()
        # print(f"{self.name} Upload Time = {end - start} sec")

    def UploadEncryt(self, gradient_dict, GlobalServer, public_key):
        #GlobalServer.gradient_buffer.clear()
        pool = Pool(POOL_CORES)
        partial_func = partial(Encrypt_Multithread, public_key=public_key)
        for layer in gradient_dict.keys():
            shape = gradient_dict[layer].shape
            gradients_array = gradient_dict[layer].cpu().numpy()
            # partial_func = partial(Encrypt_Multithread, public_key=public_key)
            encrypted_array = pool.map(partial_func, gradients_array.reshape(-1))
            encrypted_concatenate = np.concatenate(encrypted_array)
            GlobalServer.gradient_buffer[layer] = encrypted_concatenate.reshape(shape)
            print(f"Layer:{layer}, GlobalServer:{type(GlobalServer.gradient_buffer[layer])}")
        
        pool.close()
        pool.join()
        #GlobalServer.lr = self.lr
        #print(GlobalServer.gradient_buffer)

    def GenGradientsJson(self, gradient_dict, public_key, dir, file):
        pool = Pool(POOL_CORES)
        encrypted_gradient_dict = dict()
        partial_func = partial(Encrypt_Multithread, public_key=public_key)
        for layer in gradient_dict.keys():
            # print('=' * 10, f"Encrypt gradients: {layer}", '=' * 10)
            shape = gradient_dict[layer].shape
            gradients_array = gradient_dict[layer].cpu().numpy()
            # partial_func = partial(Encrypt_Multithread, public_key=public_key)
            encrypted_array = pool.map(partial_func, gradients_array.reshape(-1))
            encrypted_concatenate = np.concatenate(encrypted_array)
            encrypted_gradient_dict[layer] = encrypted_concatenate.reshape(shape)
            # print(f"Layer:{layer}, GlobalServer:{type(GlobalServer.gradient_buffer[layer])}")
        
        pool.close()
        pool.join()
        WriteGradientsJson(encrypted_gradient_dict, MIN_GRADS_EXPONENT, dir, file)

    def LocalTraining(self, GlobalServer, public_key1, private_key1, public_key2, private_key2, dir, file, noise_file, 
                        epochs = 1, budget = 0.004, isGenerate=False, isDP=False, isReturnNoise=False, isOurs=False):
        # use_cuda = torch.cuda.is_available()
        # if use_cuda:
        #     self.local_network = self.local_network.cuda()

        tmp_gradient = dict()
        # print(old_param[list(old_param.keys())[0]])
        if not isOurs:
            # print(f"Download from server")
            self.Download(GlobalServer)
        old_param = deepcopy(self.local_network.state_dict())

        for e in range(epochs):
            epoch_loss_sum = 0
            # self.DownloadDecrypt(GlobalServer, private_key1)
            for x , y in self.training_data: 
                
                if self.use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                # print('=' * 10, y.shape)
                batch_size = x.shape[0]
                x = x.view(batch_size,-1) ## CNN不用reshape # 把data tensor的形狀做改變
                net_out = self.local_network(x) # 把x丟進net去train，得到output
                loss = self.loss_fn(net_out , y) # 把train出來的結果和ground truth去算loss
                
                epoch_loss_sum += float(loss.item())

                # 做backpropagation
                self.optimizer.zero_grad() #先把optimizer清空
                loss.backward()
                #self.local_network.weight.grad , net.L3.bias.gradnet.L3.weight.grad , net.L3.bias.grad
                self.optimizer.step() # 把算完的gradient套在network的parameter

            self.loss_history.append(epoch_loss_sum)
            
        for layer in self.local_network.state_dict().keys():
            tmp_gradient[layer] = self.local_network.state_dict()[layer] - old_param[layer]
            # print(tmp_gradient[layer])

        if isDP:
            noise_dict = dict()
            for layer in tmp_gradient.keys():
                grad_square = torch.sum(tmp_gradient[layer] ** 2)
                grad_square = math.sqrt(grad_square)
                std = (1e-6)*grad_square*math.sqrt(2*math.log(1.25/1e-5))/budget
                if (math.isnan(std)):
                    std = 0
                    # print(f"square: {grad_square}")
                    # print(f"layer: {layer}")
                    # print(f"param: {tmp_gradient[layer]}")
                noise = torch.normal(0, std, size=tmp_gradient[layer].size()).to(self.device)
                if tmp_gradient[layer].dtype == torch.int64:
                    noise = noise.to(torch.int64)
                    
                # print(f" before grads: {tmp_gradient[layer]}")
                tmp_gradient[layer] += noise
                # print(f" after grads: {tmp_gradient[layer]}")
                noise_dict[layer] = noise
        
        self.Upload(tmp_gradient, GlobalServer)
        if isGenerate:
            self.GenGradientsJson(tmp_gradient, public_key1, dir, file) 
            if isDP:
                self.GenGradientsJson(noise_dict, public_key2, dir, noise_file)
        if isReturnNoise:
            return noise_dict

    def LocalTesting(self, test_dataset):
        correct_count = 0
        total_testdata = 0
        epoch_loss_sum = 0
        use_cuda = torch.cuda.is_available()
        Y = []
        Output = []

        if use_cuda:
            self.local_network = self.local_network.cuda()

        for x,y in test_dataset:
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            batch_size = x.shape[0]
            total_testdata += batch_size
            ## CNN的時候取消
            x = x.view(batch_size , -1)
            output = self.local_network(x).max(1)[1]
            net_out = self.local_network(x) 

            # validation loss
            # print(f"net_out: {net_out}")
            # print(f"y: {y}")
            loss = self.loss_fn(net_out , y)
            epoch_loss_sum += float(loss.item())

            Y = Y + y.tolist()
            Output = Output + output.tolist()
            correct_count += torch.sum(output==y).item()

        self.validate_loss.append(epoch_loss_sum / len(test_dataset))
        f1 = f1_score(Y, Output, average = "macro")
        # print('accuracy rate: ', correct_count/total_testdata)
        # print("f1-score:", f1)
        # print(f"correct count: {correct_count}, total testdata: {total_testdata}")
        return correct_count/total_testdata, f1

    # ===== 只有training, Download upload 另外呼叫 ===== #
    def OnlyTraining(self, epochs = 1): 
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.local_network = self.local_network.cuda()

        for e in range(epochs):
            epoch_loss_sum = 0

            for x , y in self.training_data: # tqdm是可以印進度條的package
                #self.Download(GlobalServer)
                #self.optimizer = torch.optim.SGD(params=self.local_network.parameters(), lr = self.lr)
                #print(GlobalServer.global_network.state_dict())
                #print(self.local_network.state_dict())
                if use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                batch_size = x.shape[0]
                ## 做CNN的時候取消
                x = x.view(batch_size,-1) # 把data tensor的形狀做改變，-1代表由pytorch決定要變多少
                net_out = self.local_network(x) # 把x丟進net去train，得到output
                loss = self.loss_fn(net_out , y) # 把train出來的結果和ground truth去算loss
                epoch_loss_sum += float(loss.item())
                # 做backpropagation
                self.optimizer.zero_grad() #先把optimizer清空
                loss.backward()
                #self.local_network.weight.grad , net.L3.bias.gradnet.L3.weight.grad , net.L3.bias.grad
                self.optimizer.step() # 把算完的gradient套在network的parameter
                ## For Cifar100
                # self.scheduler.step() ### F
                #for layer, param in zip(self.local_network.state_dict().keys(), self.local_network.parameters()):
                    #print(layer)
                    #print(param.grad)        
                #self.Upload(GlobalServer)
        
                #GlobalServer.AddGradient()
      
            self.loss_history.append(epoch_loss_sum)

    # 直接上傳參數
    def UploadParameters(self, GlobalServer):
        GlobalServer.global_network.load_state_dict(self.local_network.state_dict())

    def removenoise(self, param_dict, noiseAES_dict, cpabeAES_key, ivAES, cpabePK, cpabeMK):
        from charm.schemes.abenc.abenc_bsw07 import CPabe_BSW07
        from charm.toolbox.pairinggroup import PairingGroup
        from module.cpabe import HybridABEnc
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import unpad
        import binascii

        groupObj = PairingGroup('SS512')
        cpabe = CPabe_BSW07(groupObj)
        hyb_abe = HybridABEnc(cpabe, groupObj)
        cpabeSK = hyb_abe.keygen(cpabePK, cpabeMK, ['P1'])
        AES_key = hyb_abe.decrypt(cpabePK, cpabeSK, cpabeAES_key)
        cipher = AES.new(AES_key, AES.MODE_CBC, ivAES)

        # Decrypt AES noise
        noise_dict = dict()
        for layer in noiseAES_dict.keys():
            for i in range(len(noiseAES_dict[layer])):
                decrypt_AES = unpad(cipher.decrypt(noiseAES_dict[layer][i]), AES.block_size)
                reverseBytes = decrypt_AES[::-1]
                bytesToHex = binascii.b2a_hex(reverseBytes)
                bytesHexToASCII = bytesToHex.decode('ascii')
                noiseAES_dict[layer][i] = struct.unpack('!f', bytes.fromhex(bytesHexToASCII))[0]
                # print(noiseAES_dict[layer][i])
            # print(f"layer: {layer}")
            # print(f"noiseAES len: {len(noiseAES_dict[layer])}")
            # print(f"noiseAES: {noiseAES_dict[layer]}")
            # print(f"param shape: {param_dict[layer].shape}")
            # print(f"torch noiseAES shape: {torch.FloatTensor(noiseAES_dict[layer]).shape}")
            # print(f"torch noiseAES: {torch.FloatTensor(noiseAES_dict[layer])}")
            noise_dict[layer] = torch.FloatTensor(noiseAES_dict[layer]).reshape(param_dict[layer].shape).to(self.device)
            param_dict[layer] = param_dict[layer] - noise_dict[layer]
        
        return param_dict

def GenerateParticipant(TrainData_dict, num = 1):
    P_dict = dict()
    for i in range(num):
        s = 'P'
        s = s + str(i)
        P_dict[s] = Participant(s, TrainData_dict[s])

    return P_dict

def SplitData(train_set, test_set, num):
    train_split = int(len(train_set) / num)
    test_split = int(len(test_set) / num)

    # print(f"training data length: {len(train_set)}")
    # print(f"test data length: {len(test_set)}")
    # raise ValueError

    portions = [train_split] * num
    if (train_split * num != len(train_set)):
        portions[-1] = train_split + len(train_set) - train_split * num

    TrainSet_list = [None] * num
    TrainSet_list = data.random_split(train_set, portions)

    portions = [test_split] * num
    if (test_split * num != len(test_set)):
        portions[-1] = test_split + len(test_set) - test_split * num

    TestSet_list = [None] * num
    TestSet_list = data.random_split(test_set, portions)

    TrainDataSet_dict = dict()
    TestDataSet_dict = dict()

    # mini batch
    for i in range(num):
        s = 'P'
        s = s + str(i)
        TrainDataSet_dict[s] = data.DataLoader(dataset =  TrainSet_list[i], batch_size=64, shuffle=True)
        TestDataSet_dict[s] = data.DataLoader(dataset =  TestSet_list[i], batch_size=64, shuffle=True)

    return TrainDataSet_dict, TestDataSet_dict