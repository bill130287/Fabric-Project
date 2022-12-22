package smartcontract

import (
	"HEFL_Test/entity/system"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"math/big"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

/* For testing time */
func (s *SmartContract) OnlyUpload(ctx contractapi.TransactionContextInterface, gradsJson string, noiseJson string) error {
	log.Println("Only Upload")
	var uploadgrads []*UploadGrads
	err := json.Unmarshal([]byte(gradsJson), &uploadgrads)
	if err != nil {
		return fmt.Errorf("unmarshal error: gradsJson")
	}
	var uploadnoises []*UploadGrads
	errnoise := json.Unmarshal([]byte(noiseJson), &uploadnoises)
	if errnoise != nil {
		return fmt.Errorf("unmarshal error: noiseJson")
	}

	if upload_cnt == 0 {
		// assign gradients
		for _, layer_grads := range uploadgrads {
			log.Println("assign grads layer: ", layer_grads.LayerName)
			gradsTobigInt := system.StringTobigInt(layer_grads.Grads)
			tmp_grads[layer_grads.LayerName] = gradsTobigInt
		}
		// assign noises
		for _, layer_noises := range uploadnoises {
			log.Println("assign noises layer: ", layer_noises.LayerName)
			noisesTobigInt := system.StringTobigInt(layer_noises.Grads)
			tmp_noises[layer_noises.LayerName] = noisesTobigInt
		}
	} else {
		// multiply grads
		for _, layer_grads := range uploadgrads {
			log.Println("update grads layer: ", layer_grads.LayerName)
			gradsTobigInt := system.StringTobigInt(layer_grads.Grads)
			if len(gradsTobigInt) != len(tmp_grads[layer_grads.LayerName]) {
				log.Println("upload grads len: ", len(gradsTobigInt))
				log.Println("on chain tmp grads len: ", len(tmp_grads[layer_grads.LayerName]))
				return fmt.Errorf("the length of params and the length of uploading grads are not equal")
			}

			for i := range gradsTobigInt {
				// gradC := paillier.Cypher{C: &gradTobigInt[i]}
				// paramC := paillier.Cypher{C: &globalParameters["L1.weight"].Parameters[i]}
				accumulator := new(big.Int).Mod(
					new(big.Int).Mul(&gradsTobigInt[i], &tmp_grads[layer_grads.LayerName][i]),
					publickey1.GetNSquare(),
				)
				tmp_grads[layer_grads.LayerName][i] = *accumulator
			}
		}
		// multiply noises
		for _, layer_noises := range uploadnoises {
			log.Println("update noises layer: ", layer_noises.LayerName)
			noisesTobigInt := system.StringTobigInt(layer_noises.Grads)
			if len(noisesTobigInt) != len(tmp_noises[layer_noises.LayerName]) {
				log.Println("upload noises len: ", len(noisesTobigInt))
				log.Println("on chain tmp noises len: ", len(tmp_noises[layer_noises.LayerName]))
				return fmt.Errorf("the length of params and the length of uploading noises are not equal")
			}

			for i := range noisesTobigInt {
				accumulator := new(big.Int).Mod(
					new(big.Int).Mul(&noisesTobigInt[i], &tmp_noises[layer_noises.LayerName][i]),
					publickey2.GetNSquare(),
				)
				tmp_noises[layer_noises.LayerName][i] = *accumulator
			}
		}
	}

	upload_cnt += 1

	return nil
}

func (s *SmartContract) OnlyUpdate(ctx contractapi.TransactionContextInterface) error {
	// Aggregate
	if upload_cnt == N {
		// Aggregate grads
		log.Println("Update global model")
		for layer_key := range tmp_grads {
			log.Println("layer: ", layer_key)
			paramsJson, err := ctx.GetStub().GetState(layer_key)
			if err != nil {
				return fmt.Errorf("failed to read from world state: %v", err)
			}
			if paramsJson == nil {
				return fmt.Errorf("there is no %s layer", layer_key)
			}

			var params GlobalParameters
			err = json.Unmarshal(paramsJson, &params)
			if err != nil {
				return fmt.Errorf("failed to unmarshall paramsJson")
			}

			paramsTobigInt := system.StringTobigInt(params.Parameters)
			if len(paramsTobigInt) != len(tmp_grads[layer_key]) {
				log.Println("params len: ", len(paramsTobigInt))
				log.Println("grads len: ", len(tmp_grads[layer_key]))
				return fmt.Errorf("the length of params and the length of uploading grad are not equal")
			}

			for i := range tmp_grads[layer_key] {
				// gradC := paillier.Cypher{C: &gradTobigInt[i]}
				// paramC := paillier.Cypher{C: &globalParameters["L1.weight"].Parameters[i]}

				aggregate_grad := system.Mypow(&tmp_grads[layer_key][i], &HE_divided, publickey1.GetNSquare())

				accumulator := new(big.Int).Mod(
					new(big.Int).Mul(aggregate_grad, &paramsTobigInt[i]),
					publickey1.GetNSquare(),
				)
				if publickey1.GetNSquare().Cmp(accumulator) == -1 {
					log.Println("error layer: ", layer_key)
					log.Println("error number: ", accumulator)
					log.Println("N square: ", publickey1.GetNSquare())
				}
				params.Parameters[i] = accumulator.String()
			}
			// log.Println("new params shape: ", params.LayerShape)
			// log.Println("new params size: ", params.LayerSize)
			// log.Println("new param params: ", params.Parameters)

			newparamsJson, err := json.Marshal(params)
			if err != nil {
				return fmt.Errorf("newparamsJson failed to marshal")
			}

			// log.Println("newparamsJson: ", '\n', string(newparamsJson))
			ctx.GetStub().PutState(layer_key, newparamsJson)
		}

		// Aggregate noises
		log.Println("Update global noises")
		noise_prefix := "noise_"
		for layer_key := range tmp_noises {
			log.Println("noise layer: ", noise_prefix+layer_key)
			noisesJson, err := ctx.GetStub().GetState(noise_prefix + layer_key)
			if err != nil {
				return fmt.Errorf("failed to read from world state: %v", err)
			}
			if noisesJson == nil {
				return fmt.Errorf("there is no %s layer", layer_key)
			}

			var noises GlobalParameters
			err = json.Unmarshal(noisesJson, &noises)
			if err != nil {
				return fmt.Errorf("failed to unmarshall paramsJson")
			}

			noisesTobigInt := system.StringTobigInt(noises.Parameters)
			if len(noisesTobigInt) != len(tmp_noises[layer_key]) {
				log.Println("noises len: ", len(noisesTobigInt))
				log.Println("aggregate noises len: ", len(tmp_noises[layer_key]))
				return fmt.Errorf("the length of aggregating params and the length of uploading grad are not equal")
			}

			for i := range tmp_noises[layer_key] {
				// gradC := paillier.Cypher{C: &gradTobigInt[i]}
				// paramC := paillier.Cypher{C: &globalParameters["L1.weight"].Parameters[i]}

				aggregate_noise := system.Mypow(&tmp_noises[layer_key][i], &HE_divided, publickey2.GetNSquare())

				accumulator := new(big.Int).Mod(
					new(big.Int).Mul(aggregate_noise, &noisesTobigInt[i]),
					publickey2.GetNSquare(),
				)
				if publickey2.GetNSquare().Cmp(accumulator) == -1 {
					log.Println("error layer: ", layer_key)
					log.Println("error number: ", accumulator)
					log.Println("N square: ", publickey1.GetNSquare())
				}
				noises.Parameters[i] = accumulator.String()
			}
			// log.Println("new params shape: ", noises.LayerShape)
			// log.Println("new params size: ", noises.LayerSize)
			// log.Println("new param params: ", params.Parameters)

			newnoisesJson, err := json.Marshal(noises)
			if err != nil {
				return fmt.Errorf("newnoisesJson failed to marshal")
			}

			// log.Println("newparamsJson: ", '\n', string(newparamsJson))
			ctx.GetStub().PutState(noise_prefix+layer_key, newnoisesJson)
		}

		upload_cnt = 0
	}

	return nil
}

/* original version */
func (s *SmartContract) Upload(ctx contractapi.TransactionContextInterface, gradsJson string, noiseJson string) error {
	log.Println("function Upload")
	// log.Println("gradsjson: ", gradsJson)
	log.Println("public key nsquare: ", publickey1.GetNSquare())
	var uploadgrads []*UploadGrads
	err := json.Unmarshal([]byte(gradsJson), &uploadgrads)
	if err != nil {
		return fmt.Errorf("unmarshal error: gradsJson")
	}

	for _, _gradsJson := range uploadgrads {
		log.Println("layer: ", _gradsJson.LayerName)
		// log.Println("grads: ", _gradsJson.Grads)

		gradTobigInt := system.StringTobigInt(_gradsJson.Grads)
		paramsJson, err := ctx.GetStub().GetState(_gradsJson.LayerName)
		if err != nil {
			return fmt.Errorf("failed to read from world state: %v", err)
		}
		if paramsJson == nil {
			return fmt.Errorf("there is no %s layer", _gradsJson.LayerName)
		}

		var params GlobalParameters
		err = json.Unmarshal(paramsJson, &params)
		if err != nil {
			return fmt.Errorf("failed to unmarshall paramsJson")
		}
		log.Println("params shape: ", params.LayerShape)
		log.Println("params size: ", params.LayerSize)
		// log.Println("param params: ", params.Parameters)

		paramsTobigInt := system.StringTobigInt(params.Parameters)
		if len(paramsTobigInt) != len(gradTobigInt) {
			log.Println("params len: ", len(paramsTobigInt))
			log.Println("grads len: ", len(gradTobigInt))
			return fmt.Errorf("the length of params and the length of uploading grad are not equal")
		}

		for i := range gradTobigInt {
			// gradC := paillier.Cypher{C: &gradTobigInt[i]}
			// paramC := paillier.Cypher{C: &globalParameters["L1.weight"].Parameters[i]}
			accumulator := new(big.Int).Mod(
				new(big.Int).Mul(&gradTobigInt[i], &paramsTobigInt[i]),
				publickey1.GetNSquare(),
			)
			params.Parameters[i] = accumulator.String()
		}
		log.Println("new params shape: ", params.LayerShape)
		log.Println("new params size: ", params.LayerSize)
		// log.Println("new param params: ", params.Parameters)

		newparamsJson, err := json.Marshal(params)
		if err != nil {
			return fmt.Errorf("newparamsJson failed to marshal")
		}

		// log.Println("newparamsJson: ", '\n', string(newparamsJson))
		ctx.GetStub().PutState(_gradsJson.LayerName, newparamsJson)
	}

	// Noise Update
	var uploadnoise []*UploadGrads
	err = json.Unmarshal([]byte(noiseJson), &uploadnoise)
	if err != nil {
		return fmt.Errorf("unmarshal error: noiseJson")
	}
	prefix := "noise_"

	for _, _uploadnoise := range uploadnoise {
		log.Println("layer: ", _uploadnoise.LayerName)
		// log.Println("grads: ", _gradsJson.Grads)

		_uploadnoiseTobigInt := system.StringTobigInt(_uploadnoise.Grads)
		noiseOnChain_Json, err := ctx.GetStub().GetState(prefix + _uploadnoise.LayerName)
		if err != nil {
			return fmt.Errorf("failed to read noise from world state: %v", err)
		}
		if noiseOnChain_Json == nil {
			return fmt.Errorf("there is no %s layer of noise", _uploadnoise.LayerName)
		}

		var noiseOnChain GlobalParameters
		err = json.Unmarshal(noiseOnChain_Json, &noiseOnChain)
		if err != nil {
			return fmt.Errorf("failed to unmarshall noiseOnChainJson for layer: %s", noiseOnChain.LayerName)
		}
		// log.Println("noise shape: ", noise.LayerShape)
		// log.Println("noise size: ", noise.LayerSize)
		// log.Println("param params: ", params.Parameters)

		noiseOnChain_TobigInt := system.StringTobigInt(noiseOnChain.Parameters)
		if len(noiseOnChain_TobigInt) != len(_uploadnoiseTobigInt) {
			log.Println("noise on chain len: ", len(noiseOnChain_TobigInt))
			log.Println("noise uploaded len: ", len(_uploadnoiseTobigInt))
			return fmt.Errorf("the length of noise on chain and the length of uploading noise are not equal")
		}

		for i := range _uploadnoiseTobigInt {
			// gradC := paillier.Cypher{C: &gradTobigInt[i]}
			// paramC := paillier.Cypher{C: &globalParameters["L1.weight"].Parameters[i]}
			accumulator := new(big.Int).Mod(
				new(big.Int).Mul(&_uploadnoiseTobigInt[i], &noiseOnChain_TobigInt[i]),
				publickey1.GetNSquare(),
			)
			noiseOnChain.Parameters[i] = accumulator.String()
		}
		// log.Println("new params shape: ", params.LayerShape)
		// log.Println("new params size: ", params.LayerSize)
		// log.Println("new param params: ", params.Parameters)

		newnoiseJson, err := json.Marshal(noiseOnChain)
		if err != nil {
			return fmt.Errorf("newnoiseJson failed to marshal")
		}

		// log.Println("newparamsJson: ", '\n', string(newparamsJson))
		ctx.GetStub().PutState(prefix+noiseOnChain.LayerName, newnoiseJson)
	}

	return nil
}

// aggregate version
func (s *SmartContract) Upload2(ctx contractapi.TransactionContextInterface, gradsJson string, noiseJson string) error {
	log.Println("function Upload2")
	var uploadgrads []*UploadGrads
	err := json.Unmarshal([]byte(gradsJson), &uploadgrads)
	if err != nil {
		return fmt.Errorf("unmarshal error: gradsJson")
	}
	var uploadnoises []*UploadGrads
	errnoise := json.Unmarshal([]byte(noiseJson), &uploadnoises)
	if errnoise != nil {
		return fmt.Errorf("unmarshal error: noiseJson")
	}

	if upload_cnt == 0 {
		// assign gradients
		for _, layer_grads := range uploadgrads {
			log.Println("assign grads layer: ", layer_grads.LayerName)
			gradsTobigInt := system.StringTobigInt(layer_grads.Grads)
			tmp_grads[layer_grads.LayerName] = gradsTobigInt
		}
		// assign noises
		for _, layer_noises := range uploadnoises {
			log.Println("assign noises layer: ", layer_noises.LayerName)
			noisesTobigInt := system.StringTobigInt(layer_noises.Grads)
			tmp_noises[layer_noises.LayerName] = noisesTobigInt
		}
	} else {
		// multiply grads
		for _, layer_grads := range uploadgrads {
			log.Println("update grads layer: ", layer_grads.LayerName)
			gradsTobigInt := system.StringTobigInt(layer_grads.Grads)
			if len(gradsTobigInt) != len(tmp_grads[layer_grads.LayerName]) {
				log.Println("upload grads len: ", len(gradsTobigInt))
				log.Println("on chain tmp grads len: ", len(tmp_grads[layer_grads.LayerName]))
				return fmt.Errorf("the length of params and the length of uploading grads are not equal")
			}

			for i := range gradsTobigInt {
				// gradC := paillier.Cypher{C: &gradTobigInt[i]}
				// paramC := paillier.Cypher{C: &globalParameters["L1.weight"].Parameters[i]}
				accumulator := new(big.Int).Mod(
					new(big.Int).Mul(&gradsTobigInt[i], &tmp_grads[layer_grads.LayerName][i]),
					publickey1.GetNSquare(),
				)
				tmp_grads[layer_grads.LayerName][i] = *accumulator
			}
		}
		// multiply noises
		for _, layer_noises := range uploadnoises {
			log.Println("update noises layer: ", layer_noises.LayerName)
			noisesTobigInt := system.StringTobigInt(layer_noises.Grads)
			if len(noisesTobigInt) != len(tmp_noises[layer_noises.LayerName]) {
				log.Println("upload noises len: ", len(noisesTobigInt))
				log.Println("on chain tmp noises len: ", len(tmp_noises[layer_noises.LayerName]))
				return fmt.Errorf("the length of params and the length of uploading noises are not equal")
			}

			for i := range noisesTobigInt {
				accumulator := new(big.Int).Mod(
					new(big.Int).Mul(&noisesTobigInt[i], &tmp_noises[layer_noises.LayerName][i]),
					publickey2.GetNSquare(),
				)
				tmp_noises[layer_noises.LayerName][i] = *accumulator
			}
		}
	}

	upload_cnt += 1

	// Aggregate
	if upload_cnt == N {
		// Aggregate grads
		log.Println("Update global model")
		for _, _gradsJson := range uploadgrads {
			log.Println("layer: ", _gradsJson.LayerName)
			// log.Println("grads: ", _gradsJson.Grads)

			// gradTobigInt := system.StringTobigInt(_gradsJson.Grads)
			paramsJson, err := ctx.GetStub().GetState(_gradsJson.LayerName)
			if err != nil {
				return fmt.Errorf("failed to read from world state: %v", err)
			}
			if paramsJson == nil {
				return fmt.Errorf("there is no %s layer", _gradsJson.LayerName)
			}

			var params GlobalParameters
			err = json.Unmarshal(paramsJson, &params)
			if err != nil {
				return fmt.Errorf("failed to unmarshall paramsJson")
			}
			log.Println("params shape: ", params.LayerShape)
			log.Println("params size: ", params.LayerSize)
			// log.Println("param params: ", params.Parameters)

			paramsTobigInt := system.StringTobigInt(params.Parameters)
			if len(paramsTobigInt) != len(tmp_grads[_gradsJson.LayerName]) {
				log.Println("params len: ", len(paramsTobigInt))
				log.Println("grads len: ", len(tmp_grads[_gradsJson.LayerName]))
				return fmt.Errorf("the length of params and the length of uploading grad are not equal")
			}

			for i := range tmp_grads[_gradsJson.LayerName] {
				// gradC := paillier.Cypher{C: &gradTobigInt[i]}
				// paramC := paillier.Cypher{C: &globalParameters["L1.weight"].Parameters[i]}

				aggregate_grad := system.Mypow(&tmp_grads[_gradsJson.LayerName][i], &HE_divided, publickey1.GetNSquare())

				accumulator := new(big.Int).Mod(
					new(big.Int).Mul(aggregate_grad, &paramsTobigInt[i]),
					publickey1.GetNSquare(),
				)
				if publickey1.GetNSquare().Cmp(accumulator) == -1 {
					log.Println("error layer: ", _gradsJson.LayerName)
					log.Println("error number: ", accumulator)
					log.Println("N square: ", publickey1.GetNSquare())
				}
				params.Parameters[i] = accumulator.String()
			}
			log.Println("new params shape: ", params.LayerShape)
			log.Println("new params size: ", params.LayerSize)
			// log.Println("new param params: ", params.Parameters)

			newparamsJson, err := json.Marshal(params)
			if err != nil {
				return fmt.Errorf("newparamsJson failed to marshal")
			}

			// log.Println("newparamsJson: ", '\n', string(newparamsJson))
			ctx.GetStub().PutState(_gradsJson.LayerName, newparamsJson)
		}

		// Aggregate noises
		log.Println("Update global noises")
		noise_prefix := "noise_"
		for _, _noisesJson := range uploadnoises {
			log.Println("layer: ", noise_prefix+_noisesJson.LayerName)
			// log.Println("grads: ", _gradsJson.Grads)

			// gradTobigInt := system.StringTobigInt(_gradsJson.Grads)
			noisesJson, err := ctx.GetStub().GetState(noise_prefix + _noisesJson.LayerName)
			if err != nil {
				return fmt.Errorf("failed to read from world state: %v", err)
			}
			if noisesJson == nil {
				return fmt.Errorf("there is no %s layer", _noisesJson.LayerName)
			}

			var noises GlobalParameters
			err = json.Unmarshal(noisesJson, &noises)
			if err != nil {
				return fmt.Errorf("failed to unmarshall paramsJson")
			}
			log.Println("noises shape: ", noises.LayerShape)
			log.Println("noises size: ", noises.LayerSize)
			// log.Println("param params: ", params.Parameters)

			noisesTobigInt := system.StringTobigInt(noises.Parameters)
			if len(noisesTobigInt) != len(tmp_noises[_noisesJson.LayerName]) {
				log.Println("noises len: ", len(noisesTobigInt))
				log.Println("aggregate noises len: ", len(tmp_noises[_noisesJson.LayerName]))
				return fmt.Errorf("the length of aggregating params and the length of uploading grad are not equal")
			}

			for i := range tmp_noises[_noisesJson.LayerName] {
				// gradC := paillier.Cypher{C: &gradTobigInt[i]}
				// paramC := paillier.Cypher{C: &globalParameters["L1.weight"].Parameters[i]}

				aggregate_noise := system.Mypow(&tmp_noises[_noisesJson.LayerName][i], &HE_divided, publickey2.GetNSquare())

				accumulator := new(big.Int).Mod(
					new(big.Int).Mul(aggregate_noise, &noisesTobigInt[i]),
					publickey2.GetNSquare(),
				)
				if publickey2.GetNSquare().Cmp(accumulator) == -1 {
					log.Println("error layer: ", _noisesJson.LayerName)
					log.Println("error number: ", accumulator)
					log.Println("N square: ", publickey1.GetNSquare())
				}
				noises.Parameters[i] = accumulator.String()
			}
			log.Println("new params shape: ", noises.LayerShape)
			log.Println("new params size: ", noises.LayerSize)
			// log.Println("new param params: ", params.Parameters)

			newnoisesJson, err := json.Marshal(noises)
			if err != nil {
				return fmt.Errorf("newnoisesJson failed to marshal")
			}

			// log.Println("newparamsJson: ", '\n', string(newparamsJson))
			ctx.GetStub().PutState(noise_prefix+_noisesJson.LayerName, newnoisesJson)
		}

		upload_cnt = 0
	}

	return nil
}

func (s *SmartContract) DownloadParams(ctx contractapi.TransactionContextInterface) ([]*GlobalParameters, error) {
	log.Println("Function Download Params")

	resultsIterator, err := ctx.GetStub().GetStateByRange("", "")
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	prefix := "noise_"
	var params []*GlobalParameters
	// var totalData []*GlobalParameters
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		if !strings.Contains(queryResponse.Key, prefix) {
			log.Println("download params query response key: ", queryResponse.Key)
			var paramsJson GlobalParameters
			err = json.Unmarshal(queryResponse.Value, &paramsJson)
			if err != nil {
				return nil, err
			}
			params = append(params, &paramsJson)
		}

		if len(params) == len(tmp_grads) {
			break
		}

	}

	return params, nil
}

func (s *SmartContract) DownloadNoise(ctx contractapi.TransactionContextInterface) ([]*GlobalParameters, error) {
	log.Println("Function Download Noise")

	resultsIterator, err := ctx.GetStub().GetStateByRange("", "")
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	prefix := "noise_"
	var noise []*GlobalParameters
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		if strings.Contains(queryResponse.Key, prefix) {
			log.Println("download noise query response key: ", queryResponse.Key)
			var noisesJson GlobalParameters
			err = json.Unmarshal(queryResponse.Value, &noisesJson)
			if err != nil {
				return nil, err
			}
			noise = append(noise, &noisesJson)
		}

		if len(noise) == len(tmp_noises) {
			break
		}

	}

	return noise, nil
}

func (s *SmartContract) GetPk(ctx contractapi.TransactionContextInterface) string {
	return publickey1.N.String()
}
