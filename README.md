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
- charm-crypto (There may be problems when installing this package directly. You can try the following method to copy the environment first. If it doesn't work, you can refer to the method provided below to install it yourself.)

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
If you have any problems when building the environment, you can refer to the following information (some records I made before)

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
- After installing Hyperledger Fabric Sample (*fabric-samples*) according to the second point in [Help](#help), place *HEFL_Test*, *SDK_HEFL_Test*, and *one-click.sh* in the *FabricChain* folder under the *fabric-samples/test-network/*
- basic operation on Fabric: https://pse.is/4ledyv
