package main

import (
	"fmt"
	"github.com/r4stl1n/HalNeuralNetwork/enums"
	"github.com/r4stl1n/HalNeuralNetwork/models"
	"github.com/r4stl1n/HalNeuralNetwork/networks"
	"math/rand"
	"time"
)

func main() {

	rand.Seed(time.Now().UnixNano())

	fmt.Println("Starting this shit")

	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	expected := []float64{
		0,
		1,
		1,
		0,
	}

	gradientNetwork := networks.CreateGradientDescentNetwork("HerpDerp")

	inputLayer := models.CreateNeuralLayerFromInputs(inputs)
	hiddenLayer := models.CreateNeuralLayer(3, enums.NeuralNodeHidden, enums.NeuralActivationSigmoid)
	outputLayer := models.CreateNeuralLayer(1, enums.NeuralNodeOutput, enums.NeuralActivationSigmoid)

	inputLayerAddError := gradientNetwork.AddNeuralLayer(inputLayer)

	if inputLayerAddError != nil {
		fmt.Println(inputLayerAddError)
	}

	hiddenLayerAddError := gradientNetwork.AddNeuralLayer(hiddenLayer)

	if hiddenLayerAddError != nil {
		fmt.Println(hiddenLayerAddError)
	}

	outputLayerAddError := gradientNetwork.AddNeuralLayer(outputLayer)

	if outputLayerAddError != nil {
		fmt.Println(outputLayerAddError)
	}

	// Initialize neural connections
	createNeuralConnectionsError := gradientNetwork.CreateNeuralConnections()

	if createNeuralConnectionsError != nil {
		fmt.Println(createNeuralConnectionsError)
	}

	for i := 0; i < 15; i++ {

		_, _ = gradientNetwork.Train(inputs, expected)

	}

	for _, input := range inputs {

		_, output := gradientNetwork.ProcessData(input)

		for _, element := range output.NeuralNodes {
			fmt.Println("Input: ", input[0], input[1], " -> ", element.OutputValue)

		}
	}


	_ = expected
}
