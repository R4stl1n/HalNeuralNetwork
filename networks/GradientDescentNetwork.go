package networks

import (
	"errors"
	"fmt"
	"github.com/R4stl1n/HalNeuralNetwork/enums"
	"github.com/R4stl1n/HalNeuralNetwork/models"
	"github.com/R4stl1n/HalNeuralNetwork/util"
	uuid "github.com/satori/go.uuid"
	"math/rand"
)

type GradientDescentNetwork struct {
	UUID        string                  `json:"uuid" form:"uuid" query:"uuid"`
	NetworkName string                  `json:"network_name" form:"network_name" query:"network_name"`
	Type        enums.NeuralNetworkType `json:"network_type" form:"network_type" query:"network_type"`

	NeuralLayers      []models.NeuralLayer      `json:"neural_layers" form:"neural_layers" query:"neural_layers"`
	NeuralConnections []models.NeuralConnection `json:"neural_connections" form:"neural_connections" query:"neural_connections"`
}

func CreateGradientDescentNetwork(networkName string) *GradientDescentNetwork {

	var neuralLayers []models.NeuralLayer

	return &GradientDescentNetwork{
		UUID:         uuid.NewV4().String(),
		NetworkName:  networkName,
		Type:         enums.NeuralNetworkGradientDescent,
		NeuralLayers: neuralLayers,
	}
}

func (gradientDecentNetwork *GradientDescentNetwork) AddNeuralLayer(neuralLayer *models.NeuralLayer) error {

	// Check for empty neural layer
	if neuralLayer == nil {
		return errors.New("supplied layer is nil")
	}

	// Check to see if the first layer added is a input layer
	if len(gradientDecentNetwork.NeuralLayers) == 0 {

		if neuralLayer.Type != enums.NeuralNodeInput {
			return errors.New("first layer added was not a input layer")
		}

		neuralLayer.Index = 0

	} else {
		neuralLayer.Index = len(gradientDecentNetwork.NeuralLayers) + 1
	}

	// All good to go lets add the layer
	gradientDecentNetwork.NeuralLayers = append(gradientDecentNetwork.NeuralLayers, *neuralLayer)

	return nil
}

func (gradientDecentNetwork *GradientDescentNetwork) CreateNeuralConnections() error {

	if len(gradientDecentNetwork.NeuralLayers) <= 2 {
		return errors.New("not enough layers to create connections")
	}

	for neuralIndex, neuralElement := range gradientDecentNetwork.NeuralLayers {

		if neuralIndex == len(gradientDecentNetwork.NeuralLayers)-1 {
			// No more connections to make
		} else {

			nextLayer := gradientDecentNetwork.NeuralLayers[neuralIndex+1]

			for _, currentNode := range neuralElement.NeuralNodes {
				for _, nextNode := range nextLayer.NeuralNodes {

					gradientDecentNetwork.NeuralConnections = append(gradientDecentNetwork.NeuralConnections,
						models.NeuralConnection{
							UUID:         uuid.NewV4().String(),
							FromNodeUUID: currentNode.UUID,
							ToNodeUUID:   nextNode.UUID,
							Weight:       rand.Float64(),
						})

				}
			}
		}
	}

	return nil
}

func (gradientDecentNetwork *GradientDescentNetwork) ProcessData(data []float64) (error, *models.NeuralLayer) {

	// First check if we have at least three layers in our network
	if len(gradientDecentNetwork.NeuralLayers) < 3 {
		return errors.New("not enough layers to begin processing"), nil
	}

	// Make sure we have enough data for all the inputs

	if (gradientDecentNetwork).NeuralLayers[0].NodeCount() != len(data) {
		return errors.New("not enough input data for node amount"), nil
	}

	// Set the input data for nodes
	setInputError := (gradientDecentNetwork).NeuralLayers[0].SetInputs(data)

	if setInputError != nil {
		return setInputError, nil
	}

	// Go over each data layer and process
	for layerIndex, layerElement := range gradientDecentNetwork.NeuralLayers {

		if layerElement.Index == 0 {
			// Do nothing we need to wait for next one

		} else {
			// Network is not output

			calculateNodeOutputError := layerElement.CalculateNodeOutputs(&(gradientDecentNetwork.NeuralLayers)[layerIndex-1], gradientDecentNetwork.NeuralConnections)

			if calculateNodeOutputError != nil {
				return calculateNodeOutputError, nil
			}

			if layerElement.Type == enums.NeuralNodeOutput {
				return nil, &layerElement
			}
		}

	}

	return errors.New("no output layer found"), nil
}

func (gradientDecentNetwork *GradientDescentNetwork) Train(trainingData [][]float64, expectedResults []float64) (error, *models.NeuralLayer) {

	// First check if we have at least three layers in our network
	if len(gradientDecentNetwork.NeuralLayers) < 3 {
		return errors.New("not enough layers to begin processing"), nil
	}

	// Make sure we have enough data for all the inputs

	if (gradientDecentNetwork).NeuralLayers[0].NodeCount() != len(trainingData[0]) {
		return errors.New("not enough input data for node amount"), nil
	}

	for _, data := range trainingData {
		// Set the input data for nodes
		setInputError := gradientDecentNetwork.NeuralLayers[0].SetInputs(data)

		if setInputError != nil {
			return setInputError, nil
		}

		// Go over each data layer and process
		for layerIndex, layerElement := range gradientDecentNetwork.NeuralLayers {

			if layerElement.Index == 0 {
				// Do nothing we need to wait for next one

			} else {
				// Network is not output

				calculateNodeOutputError := layerElement.CalculateNodeOutputs(&gradientDecentNetwork.NeuralLayers[layerIndex-1], gradientDecentNetwork.NeuralConnections)

				if calculateNodeOutputError != nil {
					return calculateNodeOutputError, nil
				}

			}

		}

	}

	// Calculate the error values
	var newWeightsBackwards [][]float64
	var deltaOutputSumSaved []float64

	// Grab the list of differences
	// Iterate backwards
	for i := len(gradientDecentNetwork.NeuralLayers) - 1; i >= 1; i-- {

		var sumSigmodePrime float64
		var deltaOutputSum float64
		var deltaHiddenSum []float64

		if i >= 0 && i != 1 {

			var tempOutputWeights []float64

			currentLayer := gradientDecentNetwork.NeuralLayers[i]
			previousLayer := gradientDecentNetwork.NeuralLayers[i-1]

			for index, nodeElement := range currentLayer.NeuralNodes {

				errorValue := expectedResults[index] - nodeElement.OutputValue

				sumSigmodePrime = util.CalculateSigmoidPrime(nodeElement.BeforeActivationValue)

				deltaOutputSum = sumSigmodePrime * errorValue

				for _, previousNodeElement := range previousLayer.NeuralNodes {
					tempOutputWeights = append(tempOutputWeights, deltaOutputSum/previousNodeElement.OutputValue)
				}

				if i == len(gradientDecentNetwork.NeuralLayers)-1 {
					deltaOutputSumSaved = append(deltaOutputSumSaved, deltaOutputSum)
				}

			}

			newWeightsBackwards = append(newWeightsBackwards, tempOutputWeights)

		} else {

			// Hit the input layer do special math
			inputLayer := gradientDecentNetwork.NeuralLayers[0]
			outputLayer := gradientDecentNetwork.NeuralLayers[len(gradientDecentNetwork.NeuralLayers)-1]
			// Hidden Layer Before Output Layer
			workingLayer := gradientDecentNetwork.NeuralLayers[len(gradientDecentNetwork.NeuralLayers)-2]

			var tempValues []float64
			var tempDeriv []float64

			for _, workingNode := range workingLayer.NeuralNodes {

				for indexOutput, outputNode := range outputLayer.NeuralNodes {
					neuralConnection := models.FindNeuralConnection(workingNode.UUID, outputNode.UUID, gradientDecentNetwork.NeuralConnections)

					tempValues = append(tempValues, deltaOutputSumSaved[indexOutput]/neuralConnection.Weight)
					tempDeriv = append(tempDeriv, util.CalculateSigmoidPrime(outputNode.BeforeActivationValue))
				}
			}

			for tempIndex, tempValue := range tempValues {
				deltaHiddenSum = append(deltaHiddenSum, tempValue*tempDeriv[tempIndex])
			}

			var tempWeights []float64
			for _, deltaSum := range deltaHiddenSum {
				for _, input := range inputLayer.NeuralNodes {
					tempWeights = append(tempWeights, deltaSum/input.OutputValue)
				}
			}

			newWeightsBackwards = append(newWeightsBackwards, tempWeights)
		}
	}

	fmt.Println(util.ReverseFloat64Array(newWeightsBackwards))

	gradientDecentNetwork.setNewWeights(util.ReverseFloat64Array(newWeightsBackwards))

	return errors.New("no output layer found"), &gradientDecentNetwork.NeuralLayers[len(gradientDecentNetwork.NeuralLayers)-1]
}

func (gradientDecentNetwork *GradientDescentNetwork) setNewWeights(newWeights [][] float64) {

	weightIndexY := 0
	weightIndexX := 0
	for neuralIndex, neuralElement := range gradientDecentNetwork.NeuralLayers {

		if neuralIndex == len(gradientDecentNetwork.NeuralLayers)-1 {
			// No more connections to make
		} else {

			nextLayer := gradientDecentNetwork.NeuralLayers[neuralIndex+1]

			for _, currentNode := range neuralElement.NeuralNodes {
				for _, nextNode := range nextLayer.NeuralNodes {

					neuralConnection := models.FindNeuralConnection(currentNode.UUID, nextNode.UUID, gradientDecentNetwork.NeuralConnections)

					for neuralConnectionIndex, neuralConnectionElement := range gradientDecentNetwork.NeuralConnections {

						if neuralConnectionElement.UUID == neuralConnection.UUID {
							if weightIndexY == 0 {
								gradientDecentNetwork.NeuralConnections[neuralConnectionIndex].Weight = gradientDecentNetwork.NeuralConnections[neuralConnectionIndex].Weight - newWeights[weightIndexY][weightIndexX]
							} else {
								gradientDecentNetwork.NeuralConnections[neuralConnectionIndex].Weight = newWeights[weightIndexY][weightIndexX]
							}
							weightIndexX = weightIndexX + 1
						}
					}
				}
			}

			weightIndexY = weightIndexY + 1
			weightIndexX = 0
		}
	}

}
