import numpy as np

# paillier encrypt multithread
def Encrypt_Multithread(data_array, public_key):
    encrypt_data_array = np.array([public_key.encrypt(float(x)) for x in data_array.reshape(-1)])
    return encrypt_data_array
    #encrypted_num_list.append(encrypt_data_array)
    #print(f"Encrypt thread finishes: {i}")

# paillier decrypt multithread
def Decrypt_Multithread(encrypted_num, private_key):
    #print(f"Decrypt thread: {i}")
    # print(encrypted_num.ciphertext(False))
    decrypted_data_array = np.array([private_key.decrypt(encrypted_num)])
    return decrypted_data_array
    #decrypted_num_list.append(decrypted_data_array)
    #print(f"Decrypt thread finishes: {i}")