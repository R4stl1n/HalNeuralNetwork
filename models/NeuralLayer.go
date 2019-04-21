package models

import (
	"errors"
	"github.com/R4stl1n/HalNeuralNetwork/enums"
	uuid "github.com/satori/go.uuid"
)

type NeuralLayer struct {
	Index int `json:"index" form:"index" query:"index"`

	UUID string `json:"uuid" form:"uuid" query:"uuid"`

	Type           enums.NeuralNodeType       `json:"node_type" form:"node_type" query:"node_type"`
	ActivationType enums.NeuralActivationType `json:"activation_type" form:"activation_type" query:"activation_type"`

	NeuralNodes []NeuralNode `json:"neural_nodes" form:"neural_nodes" query:"neural_nodes"`
}

func CreateNeuralLayer(amountOfNodes int, nodeType enums.NeuralNodeType, activationType enums.NeuralActivationType) *NeuralLayer {

	var neuralNodes []NeuralNode

	for i := 0; i < amountOfNodes; i++ {
		if nodeType == enums.NeuralNodeInput {
			neuralNodes = append(neuralNodes, *CreateNeuralNode(0, 0, nodeType, activationType))

		} else {
			neuralNodes = append(neuralNodes, *CreateNeuralNode(0, 0, nodeType, activationType))
		}
	}

	return &NeuralLayer{
		UUID:           uuid.NewV4().String(),
		Type:           nodeType,
		ActivationType: activationType,
		NeuralNodes:    neuralNodes,
	}
}

func CreateNeuralLayerFromInputs(inputValues [][]float64) *NeuralLayer {

	var neuralNodes []NeuralNode

	for i := 0; i < len(inputValues[0]); i++ {
		neuralNodes = append(neuralNodes, *CreateNeuralNode(0, 0, enums.NeuralNodeInput, enums.NeuralActivationNone))
	}

	return &NeuralLayer{
		UUID:           uuid.NewV4().String(),
		Type:           enums.NeuralNodeInput,
		ActivationType: enums.NeuralActivationNone,
		NeuralNodes:    neuralNodes,
	}
}

func (neuralLayer *NeuralLayer) SetInputs(inputValues []float64) error {

	if len(inputValues) != len(neuralLayer.NeuralNodes) {
		return errors.New("not enough input values to fill nodes")
	}

	for index, inputElement := range inputValues {

		(neuralLayer.NeuralNodes)[index].OutputValue = inputElement
	}

	return nil
}

func (neuralLayer *NeuralLayer) CalculateNodeOutputs(inputLayer *NeuralLayer, neuralConnections []NeuralConnection) (error) {


	for nodeIndex, neuralNode := range neuralLayer.NeuralNodes {
		calculatedOutputError, outputValue, beforeActivationValue := neuralNode.CalculateOutput(inputLayer.NeuralNodes, neuralConnections)

		if calculatedOutputError != nil {
			return calculatedOutputError
		}

		neuralLayer.NeuralNodes[nodeIndex].OutputValue = outputValue
		neuralLayer.NeuralNodes[nodeIndex].BeforeActivationValue = beforeActivationValue
	}

	return nil

}

func (neuralLayer *NeuralLayer) NodeCount() int {
	return len(neuralLayer.NeuralNodes)
}
