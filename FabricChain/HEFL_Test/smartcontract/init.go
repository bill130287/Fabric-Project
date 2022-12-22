package smartcontract

import (
	"HEFL_Test/entity/paillier"
	"encoding/json"
	"fmt"
	"log"
	"math/big"
	"strconv"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// SmartContract provides functions for transferring tokens between accounts
type SmartContract struct {
	contractapi.Contract
}

type GlobalParameters struct {
	LayerName  string   `json:"layername"`
	LayerShape string   `json:"layershape"`
	LayerSize  string   `json:"layersize"`
	Parameters []string `json:"parameters"`
}

type UploadGrads struct {
	LayerName string   `json:"layer"`
	Grads     []string `json:"gradients"`
}

// var globalParameters = make(map[string]GlobalParameters)
var publickey1 paillier.PublicKey
var publickey2 paillier.PublicKey
var tmp_grads = make(map[string][]big.Int)
var tmp_noises = make(map[string][]big.Int)
var layers []string
var upload_cnt int
var HE_divided big.Int

var N int // aggregate number

// func (s *SmartContract) Init(ctx contractapi.TransactionContextInterface, paramsJson string, noiseJson string, pk1 string, pk2 string, _N string) error
// func (s *SmartContract) Init(ctx contractapi.TransactionContextInterface, paramsJson string, pk1 string, pk2 string, _N string) error {
func (s *SmartContract) Init(ctx contractapi.TransactionContextInterface, paramsJson string, noiseJson string, pk1 string, pk2 string, _N string) error {
	log.Println("Into init")
	upload_cnt = 0
	N, _ = strconv.Atoi(_N)
	log.Println("N: ", N)

	if N == 5 {
		tmp, err_tmp := new(big.Int).SetString("14411518807585588", 10)
		if !err_tmp {
			return fmt.Errorf("set HE_divided string error")
		}
		HE_divided = *tmp
	} else if N == 10 {
		tmp, err_tmp := new(big.Int).SetString("7205759403792794", 10)
		if !err_tmp {
			return fmt.Errorf("set HE_divided string error")
		}
		HE_divided = *tmp
	} else if N == 15 {
		tmp, err_tmp := new(big.Int).SetString("4803839602528529", 10)
		if !err_tmp {
			return fmt.Errorf("set HE_divided string error")
		}
		HE_divided = *tmp
	}
	log.Println("HE_divided: ", HE_divided)

	var uploadparams []*GlobalParameters
	err := json.Unmarshal([]byte(paramsJson), &uploadparams)
	if err != nil {
		return fmt.Errorf("unmarshal error: paramsJson")
	}

	// noise 上傳格式和params上傳格式相同 所以這邊共用同個structure
	var noise []*GlobalParameters
	err_noise := json.Unmarshal([]byte(noiseJson), &noise)
	if err_noise != nil {
		return fmt.Errorf("unmarshal error: noiseJson")
	}

	keybigInt := new(big.Int)
	keybigInt, err2 := keybigInt.SetString(pk1, 10)
	if !err2 {
		log.Printf("stringtobigInt error 1")
	}

	publickey1 = paillier.PublicKey{N: keybigInt}
	log.Println("pk 1: ", publickey1.N)

	keybigInt = new(big.Int)
	keybigInt, err2 = keybigInt.SetString(pk2, 10)
	if !err2 {
		log.Printf("stringtobigInt error 2")
	}

	publickey2 = paillier.PublicKey{N: keybigInt}
	log.Println("pk 2: ", publickey2.N)

	layers = layers[:0]
	for _, _uploadparam := range uploadparams {
		log.Println("layer: ", _uploadparam.LayerName)
		log.Println("shape: ", _uploadparam.LayerShape)
		log.Println("size: ", _uploadparam.LayerSize)

		layers = append(layers, _uploadparam.LayerName)

		paramJson, err := json.Marshal(_uploadparam)
		if err != nil {
			return fmt.Errorf("newparamsJson failed to marshal")
		}

		// log.Println("newparamsJson: ", '\n', string(paramJson))
		ctx.GetStub().PutState(_uploadparam.LayerName, paramJson)
	}

	prefix := "noise_"
	for _, _initNoise := range noise {
		log.Println("noise layer: ", _initNoise.LayerName)
		log.Println("noise shape: ", _initNoise.LayerShape)
		log.Println("noise size: ", _initNoise.LayerSize)

		noiseJson, err := json.Marshal(_initNoise)
		if err != nil {
			return fmt.Errorf("newparamsJson failed to marshal")
		}

		// log.Println("newparamsJson: ", '\n', string(paramJson))
		ctx.GetStub().PutState(prefix+_initNoise.LayerName, noiseJson)

	}

	return nil
}
