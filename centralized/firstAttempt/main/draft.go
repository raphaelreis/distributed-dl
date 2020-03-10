package main


import (

	"fmt"

	"gonum.org/v1/gonum/mat"
	
)

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
   }

func main() {

	v := []float64{1,2,3,4,5,6,7,8,9,10,11,12}
	A := mat.NewDense(3, 4, v)
	B := mat.NewDense(4, 4, make([]float64, len(v)))
	matPrint(A)
	matPrint(B)
	// B.Mul(A.T(), A)
	// matPrint(B)
	

}