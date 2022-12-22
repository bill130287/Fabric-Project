package main

import (
	"HEFL_Test/smartcontract"
	"log"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

func main() {
	tokenChaincode, err := contractapi.NewChaincode(&smartcontract.SmartContract{})
	if err != nil {
		log.Panicf("Error creating HE-FLtest chaincode: %v", err)
	}

	if err := tokenChaincode.Start(); err != nil {
		log.Panicf("Error starting HE-FLtest chaincode: %v", err)
	}
}
