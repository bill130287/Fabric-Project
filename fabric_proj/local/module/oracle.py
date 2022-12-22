from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from charm.toolbox.pairinggroup import PairingGroup
from charm.schemes.abenc.abenc_bsw07 import CPabe_BSW07

import struct
from module.cpabe import HybridABEnc
from module.util import DecryptDownloadParameters
import time

class Oracle():
    def __init__(self, exponent, pk, sk, device, main_dir = "./ExperimentFile/1" + f"/in_clients_5" + f"/CIFAR10" + f"/round{0 + 1}" + "/download_noises.json") -> None:
        self.noise_maindir = main_dir
        self.exponent = exponent
        self.pk = pk
        self.sk = sk
        self.device = device

    def handlenoises(self, subdir):
        noisepath = self.noise_maindir + subdir
        # print(noisepath)
        start = time.time()
        noise_dict = DecryptDownloadParameters(noisepath, self.exponent, self.pk, self.sk, self.device)
        end = time.time()
        print(f"decrypt noise time: {end - start}")
        # noise_dict = dict()
        # noise_dict["test"] = torch.FloatTensor([[0.25, 0.28, 0.7], [0.3, 0.1, 0.9]]).to(self.device)
        
        start = time.time()
        salt = get_random_bytes(32)
        mypassword = "AESOracle"
        key = PBKDF2(mypassword, salt, dkLen=32)
        cipher = AES.new(key, AES.MODE_CBC)

        AES_dict = dict()
        for layer in noise_dict.keys():
            AES_dict[layer] = list() 
            tmp_noise = noise_dict[layer].reshape(-1)
            for i in range(len(tmp_noise)):
                AES_dict[layer].append(cipher.encrypt(pad(struct.pack('f', tmp_noise[i]), AES.block_size)))
        
        AES_dict, AES_key, ivAES = self._AESEncrypt(noise_dict)
        # print(type(AES_key))
        # print(AES_dict)

        cpabePK, cpabeMK, cpabeAES_key = self._CPABEEncrypt(AES_key)
        end = time.time()
        print(f"AES encrypt time: {end - start}")
        return AES_dict, cpabeAES_key, ivAES, cpabePK, cpabeMK

    def handlenoises_dict(self, noise_dict):
        salt = get_random_bytes(32)
        mypassword = "AESOracle"
        key = PBKDF2(mypassword, salt, dkLen=32)
        cipher = AES.new(key, AES.MODE_CBC)

        AES_dict = dict()
        for layer in noise_dict.keys():
            AES_dict[layer] = list() 
            tmp_noise = noise_dict[layer].reshape(-1)
            for i in range(len(tmp_noise)):
                AES_dict[layer].append(cipher.encrypt(pad(struct.pack('f', tmp_noise[i]), AES.block_size)))
        
        AES_dict, AES_key, ivAES = self._AESEncrypt(noise_dict)
        # print(type(AES_key))
        # print(AES_dict)

        cpabePK, cpabeMK, cpabeAES_key = self._CPABEEncrypt(AES_key)
        
        return AES_dict, cpabeAES_key, ivAES, cpabePK, cpabeMK

    def _AESEncrypt(self, noise_dict):
        salt = get_random_bytes(32)
        mypassword = "AESOracle"
        key = PBKDF2(mypassword, salt, dkLen=32)
        cipher = AES.new(key, AES.MODE_CBC)

        AES_dict = dict()
        for layer in noise_dict.keys():
            AES_dict[layer] = list() 
            tmp_noise = noise_dict[layer].reshape(-1)
            for i in range(len(tmp_noise)):
                AES_dict[layer].append(cipher.encrypt(pad(struct.pack('f', tmp_noise[i]), AES.block_size)))
        
        return AES_dict, key, cipher.iv
    
    def _CPABEEncrypt(self, AES_key):
        groupObj = PairingGroup('SS512')
        cpabe = CPabe_BSW07(groupObj)
        hyb_abe = HybridABEnc(cpabe, groupObj)

        access_policy = '('
        for i in range (15):
            access_policy += f"P{i}"
            if i == 14:
                break
            access_policy += f" or "
        access_policy += ')'
       
        # message = b"hello world this is an important message."
        (pk, mk) = hyb_abe.setup()
        # sk = hyb_abe.keygen(pk, mk, ['ONE', 'TWO', 'THREE'])
        
        ct = hyb_abe.encrypt(pk, AES_key, access_policy)
        # mdec = hyb_abe.decrypt(pk, sk, ct)
        # assert mdec == message, "Failed Decryption!!!"
        
        return pk, mk, ct

if __name__ == '__main__':
    # oracle = Oracle(0,0,0,"cuda")
    # oracle.handlenoises()
    access_policy = '('
    for i in range (15):
        access_policy += f"P{i}"
        if i == 14:
            break
        access_policy += f" or "
    access_policy += ')'

    print(access_policy)