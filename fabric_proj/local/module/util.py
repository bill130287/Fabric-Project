from phe import paillier
import json
from multiprocessing import Pool
from functools import partial
import numpy as np
import torch
import os 
import torchvision.transforms as transforms
import torchvision.datasets as dset 

from module.multithread_crypto import Decrypt_Multithread

POOL_CORES = 16
"""
將下載參數的json檔解密拿來使用
"""
# 從下載的json檔中，從字串reconstruct成shape tuple
def GetShapeTuple(shape_str):
    shape_list = shape_str.split(',')
    for i in range(len(shape_list)):
        if shape_list[i] != '':
            shape_list[i] = int(shape_list[i])
    if shape_list[-1] == '':
        return (shape_list[0], )
    else:
        return tuple(shape_list)

# 從下載的json檔中，將加密數值先建成paillier.EncryptedNumber物件
# 再將其解密回實際數值
def DecryptDownloadParameters(path, exponent, public_key, private_key, device):
    param_dict = dict()
    with open(path, 'r') as f:
        download_str = f.readlines()
    
    param_str = download_str[0][1:-1].split("},")
    print(f"param str len: {len(param_str)}")

    pool = Pool(POOL_CORES)
    partial_func = partial(Decrypt_Multithread, private_key=private_key)
    for i in range(len(param_str)):
        if i != len(param_str) - 1:
            tmp_dict = json.loads(param_str[i] + '}')
        else:
            tmp_dict = json.loads(param_str[i])
        
        layername = tmp_dict["layername"]
        # print(f"layer: {layername}")
        shape = GetShapeTuple(tmp_dict["layershape"][1:-1])

        # print(f"layer: {layername}")
        # print(f"shape: {shape}, type: {type(shape)}, shape 0: {shape[0]}")
        # print(isinstance(shape[0], int))

        for j in range(len(tmp_dict["parameters"])):
            tmp_dict["parameters"][j] = paillier.EncryptedNumber(public_key, int(tmp_dict["parameters"][j]), exponent)
            # print(f"{j}: {private_key.decrypt(tmp_dict['parameters'][j])}")
            # if (private_key.decrypt(tmp_dict["parameters"][j]) > 1e9):
            #     print(f"j: {j}")
            #     print("ciphertext: ", tmp_dict["parameters"][j].ciphertext(False))
           
        # pool = Pool(POOL_CORES)
        # partial_func = partial(Decrypt_Multithread, private_key=private_key)
        decrypted_array = pool.map(partial_func, tmp_dict["parameters"])
        if isinstance(shape[0], int):
            param_dict[layername] = torch.Tensor(np.concatenate(decrypted_array).reshape(shape)).to(device)
        else:
            param_dict[layername] = torch.Tensor(np.concatenate(decrypted_array).reshape(())).to(device)

    pool.close()
    pool.join()
    return param_dict


"""
全域參數加密 輸出成json檔
"""
# 將加密參數的exponent decrease到指定的exponent
def DecreaseCiphertextExponent(network_dict, exponent):
    output_dict = dict()
    shape_dict = dict()
    for key in network_dict.keys():
        # print(f"key:{key}")
        #print(Server.global_network_dict[key].shape)
        tmp_params = network_dict[key]
        layer_shape = tmp_params.shape
        shape_dict[key] = layer_shape
        tmp_params = tmp_params.reshape(-1)
        for i, cipher_params in enumerate(tmp_params):
            tmp_params[i] = cipher_params.decrease_exponent_to(exponent)
        
        output_dict[key] = tmp_params
    
    return output_dict, shape_dict

# 將加密的模型參數輸出成json檔
def WriteParametersJson(output_dict, shape_dict, exponent, dir, file):
    if not os.path.exists(dir):
        os.mkdir(dir)
        
    path = dir + file
    with open(path, "w") as f:
        f.write("{\n")
        f.write(" " * 4 + "\"min exponent\": [ " + f"{exponent} ],\n")
        f.write(" " * 4 + "\"parameters\": [\n")
        for i, key in enumerate(output_dict.keys()):
            f.write(" " * 8 + "{\n")
            f.write(" " * 12 + "\"layername\": " + f"\"{key}\"" + ",\n")
            f.write(" " * 12 + "\"layershape\": " + f"\"{str(shape_dict[key])}\"" + ",\n")
            f.write(" " * 12 + "\"layersize\": " + f"\"{str(len(output_dict[key]))}\"" + ",\n")
            f.write(" " * 12 + "\"parameters\": [\n")
            f.write(" " * 16)
            l = output_dict[key].shape[0]
            for j, c in enumerate(output_dict[key]):
                if j != len(output_dict[key]) - 1: # j != len(output_dict[key]) - 1
                    f.write(f"\"{c.ciphertext(False)}\", ")
                else:
                    f.write(f"\"{c.ciphertext(False)}\"\n")
                # if j == 9: # test only write ten params
                #     break
            f.write(" " * 12 + "]\n")
            if i != len(output_dict.keys()) - 1:
                f.write(" " * 8 + "},\n")
            else:
                f.write(" " * 8 + "}\n")

        f.write(" " * 4 + "]\n")
        f.write("}")
    f.close()

"""
上傳的 gradients 輸出成Json
"""
def WriteGradientsJson(gradients_dict, min_exponent, dir, file):
    # min_exponent = GetMinExponent(gradient_dict)
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    path = dir + file
    decrease_grdients_dict, _ = DecreaseCiphertextExponent(gradients_dict, min_exponent)
    with open(path, "w") as f:
        f.write("{\n")
        f.write(" " * 4 + "\"min exponent\": [ " + f"{min_exponent} ],\n")
        f.write(" " * 4 + "\"grads\": [\n")
        for i, key in enumerate(decrease_grdients_dict.keys()):
            f.write(" " * 8 + "{\n")
            f.write(" " * 12 + "\"layer\": " + f"\"{key}\"" + ",\n")
            # f.write(" " * 12 + "\"layershape\": " + f"\"{str(shape_dict[key])}\"" + ",\n")
            # f.write(" " * 12 + "\"layersize\": " + f"\"{str(len(decrease_grdients_dict[key]))}\"" + ",\n")
            f.write(" " * 12 + "\"gradients\": [\n")
            f.write(" " * 16)
            l = decrease_grdients_dict[key].shape[0]
            for j, c in enumerate(decrease_grdients_dict[key]):
                if j != len(decrease_grdients_dict[key]) - 1: # j != len(output_dict[key]) - 1
                    f.write(f"\"{c.ciphertext(False)}\", ")
                else:
                    f.write(f"\"{c.ciphertext(False)}\"\n")
                # if j == 9: # test only write ten params
                #     break
            f.write(" " * 12 + "]\n")
            if i != len(decrease_grdients_dict.keys()) - 1:
                f.write(" " * 8 + "},\n")
            else:
                f.write(" " * 8 + "}\n")

        f.write(" " * 4 + "]\n")
        f.write("}")
    f.close()


def preprocessDataset(dataset="MNIST"):
    if dataset == "MNIST":
        trans = transforms.Compose([transforms.ToTensor()]) 
        train_set = dset.MNIST(root='../Dataset', train=True, download=True, transform=trans)
        test_set = dset.MNIST(root='../Dataset', train=False, transform=trans)
    elif dataset == "CIFAR10":
        trans = transforms.Compose([
                    #將圖片尺寸resize到32x32
                    transforms.Resize((32,32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        train_set = dset.CIFAR10(root='../Dataset', train=True, download=True ,transform=trans)
        test_set = dset.CIFAR10(root='../Dataset', train=False,transform=trans)
    elif dataset == "CIFAR100":
        trans = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.8),
                    transforms.CenterCrop(32),
                    transforms.RandomCrop(32, padding=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[n/255.
                        for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])
                ])
        train_set = dset.CIFAR100(root='../Dataset', train=True, download=True ,transform=trans)
        test_set = dset.CIFAR100(root='../Dataset', train=False,transform=trans)
    else:
        assert(f"dataset {dataset} not exist.")
    
    return train_set, test_set

def WriteResult(experiment, accu_list, loss_list):
    import csv
    if not os.path.exists('./result'):
        os.makedirs('./result') 

    with open("./result/recordnew.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([experiment])
        writer.writerow(accu_list)
        writer.writerow(loss_list)
        # f.write(f"{experiment}:\n")
        # for x in accu_list:
        #     f.write(f"{x}, ")
        # f.write('\n')