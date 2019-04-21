package util

import "math"

func CalculateSigmoid(input float64) float64 {
	return 1 / (1 + math.Exp(input))
}

func CalculateSigmoidPrime(input float64) float64 {
	return CalculateSigmoid(input) * (1 - CalculateSigmoid(input))
}

func CalculateSigmoidTransferDerivative(input float64) float64 {
	return input * (1.0 - input)
}

func CalculateRelu(input float64) float64 {
	return math.Max(0, input)
}

func ReverseFloat64Array(arrayToReverse [][]float64) [][]float64 {

	newArray := make([][]float64, len(arrayToReverse))
	for i, j := 0, len(arrayToReverse)-1; i < j; i, j = i+1, j-1 {
		newArray[i], newArray[j] = arrayToReverse[j], arrayToReverse[i]
	}
	return newArray

}
