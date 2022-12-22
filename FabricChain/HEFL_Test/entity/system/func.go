package system

import (
	"HEFL_Test/entity/paillier"
	"fmt"
	"math/big"
)

func StringTobigInt(paramString []string) []big.Int {
	var parambigInt []big.Int = make([]big.Int, len(paramString))
	for i := 0; i < len(paramString); i++ {
		tmp := new(big.Int)
		param, err := tmp.SetString(paramString[i], 10)
		if !err {
			fmt.Println("stringtobigInt error")
		}
		parambigInt[i] = *param
	}
	return parambigInt
}

func Add(Pk *paillier.PublicKey, cypher ...*big.Int) *big.Int {
	accumulator := big.NewInt(1)

	for _, c := range cypher {
		accumulator = new(big.Int).Mod(
			new(big.Int).Mul(accumulator, c),
			Pk.GetNSquare(),
		)
	}

	return accumulator
}

func Mypow(x *big.Int, y *big.Int, mod *big.Int) *big.Int {
	num := big.NewInt(1)
	_y, _ := new(big.Int).SetString(y.String(), 10)
	_x, _ := new(big.Int).SetString(x.String(), 10)

	for _y.Cmp(big.NewInt(0)) == 1 {
		if _y.Bit(0) == 1 {
			num.Mul(num, _x)
			num.Mod(num, mod)
		}
		_y.Rsh(_y, 1)
		_x.Mul(_x, _x)
		_x.Mod(_x, mod)
	}

	return num
}

// func Mypow(x *big.Int, y *big.Int, mod *big.Int) *big.Int {
// 	num := big.NewInt(1)
// 	for y.Cmp(big.NewInt(0)) == 1 {
// 		if y.Bit(0) == 1 {
// 			num.Mul(num, x)
// 			num.Mod(num, mod)
// 		}
// 		y.Rsh(y, 1)
// 		x.Mul(x, x)
// 		x.Mod(x, mod)
// 	}

// 	return num
// }
