# fabric-network
hyperledger fabric 2.0 network node sdk sample

Using fabric-network interactive with hyperledger fabric 2.0 chaincode.

# create fabric-sample test-network
https://hyperledger-fabric.readthedocs.io/en/release-2.3/test_network.html


# copy config file from test-network

``` bash
# user public key
cp <fabric-samples path>/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp/signcerts/Admin@org1.example.com-cert.pem  <project path>/metadata/
# user private key
cp <fabric-samples>/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp/keystore/priv_sk  <project path>/metadata/
# connection file
cp <fabric-samples>/test-network/organizations/peerOrganizations/org1.example.com/connection-org1.json <project path>/metadata/

```
# edit ./src/config.ts file
sample:
``` node
export const config = {
  mspId: 'Org1MSP', 
  identity: 'Admin@org1.example.com',
  channelName: 'users',
  chaincodeId: 'users',
  certPath: '../metadata/Admin@org1.example.com-cert.pem',
  privPath: '../metadata/priv_sk',
  connectionProfilePath: '../metadata/connection-org1.json'
}

```

# run
```
npm i
npm run dev
```