package models

import (
	"errors"
	"github.com/r4stl1n/HalNeuralNetwork/enums"
	"github.com/r4stl1n/HalNeuralNetwork/util"
	uuid "github.com/satori/go.uuid"
)

type NeuralNode struct {
	UUID string `json:"uuid" form:"uuid" query:"uuid"`

	Bias                  float64 `json:"bias" form:"bias" query:"bias"`
	OutputValue           float64 `json:"output_value" form:"output_value" query:"output_value"`
	BeforeActivationValue float64 `json:"before_activation_Value" form:"before_activation_Value" query:"before_activation_Value"`

	Type           enums.NeuralNodeType       `json:"node_type" form:"node_type" query:"node_type"`
	ActivationType enums.NeuralActivationType `json:"activation_type" form:"activation_type" query:"activation_type"`

	ErrorDelta float64 `json:"error_delta" form:"error_delta" query:"error_delta"`
}

func CreateNeuralNode(outputValue float64, bias float64, nodeType enums.NeuralNodeType, activationType enums.NeuralActivationType) *NeuralNode {

	return &NeuralNode{
		UUID:           uuid.NewV4().String(),
		Bias:           bias,
		OutputValue:    outputValue,
		Type:           nodeType,
		ActivationType: activationType,
	}
}

func (neuralNode *NeuralNode) CalculateOutput(inputNodes []NeuralNode, neuralConnections []NeuralConnection) (error, float64, float64) {

	var calculatedOutput float64
	var beforeActivationValue float64

	calculatedOutput = 0
	beforeActivationValue = 0

	// Check to make sure we have at least one connecting node
	if len(inputNodes) >= 1 {

		for _, element := range inputNodes {
			neuralConnection := FindNeuralConnection(element.UUID, neuralNode.UUID, neuralConnections)
			calculatedOutput = calculatedOutput + (element.OutputValue * neuralConnection.Weight)
		}

		calculatedOutput = calculatedOutput + neuralNode.Bias

		beforeActivationValue = calculatedOutput

		if neuralNode.ActivationType == enums.NeuralActivationSigmoid {
			calculatedOutput = util.CalculateSigmoid(calculatedOutput)
		} else if neuralNode.ActivationType == enums.NeuralActivationRelu {
			calculatedOutput = util.CalculateRelu(calculatedOutput)
		} else if neuralNode.ActivationType == enums.NeuralActivationSigmoidTransfer {
			calculatedOutput = util.CalculateSigmoidTransferDerivative(calculatedOutput)
		}

	} else {
		return errors.New("could not calculate output, no nodes present"), 0.0, 0.0
	}

	neuralNode.OutputValue = calculatedOutput
	neuralNode.BeforeActivationValue = beforeActivationValue

	return nil, calculatedOutput, beforeActivationValue
}
