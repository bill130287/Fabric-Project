# Federated Learning on Hyperledger Fabric
FL + CPABE + Paillier encryption on Hyperledger Fabric

# Directory Structure  
```
.  
├── FabricChain  
│   ├── HEFL_Test (chaincode)  
│   ├── SDK_HEFL_Test (NodeJs SDK for Fabric)  
│   ├── hyperledger.yaml (create environment in Anaconda)  
│   └── one-click.sh (create test-network and deploy chaincode on Fabric)    
├── fabric_proj  
│   ├── local (for local training) 
│   └── charm.yaml (create environment in Anaconda)    
├── LICENSE  
└── README.md  
```

# Create Environment
- install ***Anaconda*** first

## for local training (charm-crypto)
- modify "prefix" in charm.yaml first 
- charm-crypto 這個套件直接安裝有可能會有問題，可以先試用下面的方法複製環境，不行的話可參考下方提供的方法自行安裝

```bash
cd fabric_proj/
conda env create -f charm.yaml 
```

## for Hyperledger
- modify "prefix" in hyperledger.yaml first

```bash
cd FabricChain/
conda env create -f hyperledger.yaml 
```
# Help
建環境時若有問題，可以參考以下資料(我之前做的一點紀錄)

- install ***charm-crypto***: https://pse.is/4nx7kz
- install ***docker*** and test ***Hyperledger Fabric Sample***: https://pse.is/4p2l5x

# Usage
## Local  (Python)
```bash
cd fabric_proj/local

# training
python main.py --dataset MNIST --inclients 5 --budget 4e-3 --mode 1

# testing (ours)
python main.py --test
```
## Fabric
- 根據 Help 中第二點安裝完 Hyperledger Fabric Sample (fabric-samples) 後，將FabricChain資料夾下的 HEFL_Test、SDK_HEFL_Test、one-click.sh，放在 fabric-samples/test-network/ 路徑下
- basic operation on Fabric: https://pse.is/4ledyv
