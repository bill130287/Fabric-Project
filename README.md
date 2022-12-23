# Federated Learning on Hyperledger Fabric
FL + CPABE + Paillier encryption on Hyperledger Fabric

# Directory Structure  
```
.  
├── FabricChain  
│   ├── HEFL_Test (chaincode)  
│   ├── SDK (NodeJs SDK for Fabric)  
│   ├── hyperledeger.yaml (create environment in Anaconda)  
│   └── one-click.sh (create test-network and deploy chaincode on Fabric)    
├── fabric_proj  
│   ├── local (for local training) 
│   └── charm.yaml (create environment in Anaconda)    
├── LICENSE  
└── README.md  
```

# Create Environment
## for local training (charm-crypto)
```bash
# ######################################################################################################### #
# - modify "prefix" in charm.yaml                                                                          #
# - charm-crypto 這個套件直接安裝有可能會有問題，可以先試用下面的方法複製環境，不行的話可參考下方提供的方法自行安裝  #
# ######################################################################################################### #

cd fabric_proj/
conda env create -f charm.yaml 
```
## for Hyperledger                                                                              #
```bash
# ###################################### #
# - modify "prefix" in hyperledger.yaml  #
# ###################################### #
cd FabricChain/
conda env create -f hyperledger.yaml 
```
