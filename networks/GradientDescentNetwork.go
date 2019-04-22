package networks

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/r4stl1n/HalNeuralNetwork/enums"
	"github.com/r4stl1n/HalNeuralNetwork/models"
	"github.com/r4stl1n/HalNeuralNetwork/util"
	uuid "github.com/satori/go.uuid"
	"io/ioutil"
	"math/rand"
	"os"
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
		neuralLayer.Index = len(gradientDecentNetwork.NeuralLayers)
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

	outputLayer := gradientDecentNetwork.NeuralLayers[len(gradientDecentNetwork.NeuralLayers)-1]


	fmt.Println("OUTPUT: ", outputLayer.NeuralNodes[0].OutputValue)

	var newWeightsBackwards [][]float64
	var outputDeltaOutputSum float64
	var deltaWeightsFull []float64

	for i := len(gradientDecentNetwork.NeuralLayers) - 1; i >= 1; i-- {

		currentLayer := gradientDecentNetwork.NeuralLayers[i]
		previousLayer := gradientDecentNetwork.NeuralLayers[i-1]

		var tempOutputWeights []float64

		if i >= 0 && i != 1 {

			for index, nodeElement := range currentLayer.NeuralNodes {

				errorValue := expectedResults[index] - nodeElement.OutputValue

				fmt.Println("ERRVAL: ", errorValue)
				sumSigmodePrime := util.CalculateSigmoidPrime(nodeElement.BeforeActivationValue)

				deltaOutputSum := sumSigmodePrime * errorValue

				fmt.Println("DELTAOUTPUTSUM: ", deltaOutputSum)

				for previousNodeIndex, previousNodeElement := range previousLayer.NeuralNodes {
					tempOutputWeights = append(tempOutputWeights, deltaOutputSum/previousNodeElement.OutputValue)
					fmt.Println("HR", previousNodeIndex, ": ", previousNodeElement.OutputValue)
				}

				if nodeElement.Type == enums.NeuralNodeOutput {
					outputDeltaOutputSum = deltaOutputSum
				}
			}

			newWeightsBackwards = append(newWeightsBackwards, tempOutputWeights)

		} else {
			// Hit the input layer do special math
			inputLayer := gradientDecentNetwork.NeuralLayers[0]
			outputLayer := gradientDecentNetwork.NeuralLayers[len(gradientDecentNetwork.NeuralLayers)-1]
			hiddenBeforeLast := gradientDecentNetwork.NeuralLayers[len(gradientDecentNetwork.NeuralLayers)-2]
			// Hidden Layer Before Output Layer

			var deltaHiddenSumArray []float64

			for _, hiddenBeforeLastElement := range hiddenBeforeLast.NeuralNodes {

				for _, outputLayerElement := range outputLayer.NeuralNodes {
					neuralConnection := models.FindNeuralConnection(hiddenBeforeLastElement.UUID, outputLayerElement.UUID, gradientDecentNetwork.NeuralConnections)

					deltaHiddenSumArray = append(deltaHiddenSumArray, (outputDeltaOutputSum/neuralConnection.Weight)*util.CalculateSigmoidPrime(hiddenBeforeLastElement.BeforeActivationValue))

				}
			}

			var deltaWeights []float64

			for _, inputLayerElement := range inputLayer.NeuralNodes {
				for _, deltaHiddenSumElement := range deltaHiddenSumArray {

					deltaWeights = append(deltaWeights, deltaHiddenSumElement / inputLayerElement.OutputValue)

				}
			}

			fmt.Println("DW: ",deltaWeights)
			deltaWeightsFull = deltaWeights
		}

	}

	var newFullWeights [][]float64


	newFullWeights = append(newFullWeights, deltaWeightsFull)
	newFullWeights = append(newFullWeights, newWeightsBackwards...)

	fmt.Println("New Deltas: ",newFullWeights)
	gradientDecentNetwork.processWeightDeltas(newFullWeights)


	return errors.New("no output layer found"), &gradientDecentNetwork.NeuralLayers[len(gradientDecentNetwork.NeuralLayers)-1]
}

func (gradientDecentNetwork *GradientDescentNetwork) setNewWeights() {

	for neuralIndex, neuralElement := range gradientDecentNetwork.NeuralLayers {

		if neuralIndex == 0 {

		} else {

			previousLayer := gradientDecentNetwork.NeuralLayers[neuralIndex-1]

			for _, currentNode := range neuralElement.NeuralNodes {

				for _, previousLayerElement := range previousLayer.NeuralNodes {

					neuralConnection := models.FindNeuralConnection(previousLayerElement.UUID, currentNode.UUID, gradientDecentNetwork.NeuralConnections)

					for neuralConnectionIndex, neuralConnectionElement := range gradientDecentNetwork.NeuralConnections {

						if neuralConnectionElement.UUID == neuralConnection.UUID {

							gradientDecentNetwork.NeuralConnections[neuralConnectionIndex].Weight = gradientDecentNetwork.NeuralConnections[neuralConnectionIndex].Weight + (0.1 * currentNode.ErrorDelta * previousLayerElement.OutputValue)

						}
					}

				}

				neuralConnection := models.FindNeuralConnection(previousLayer.NeuralNodes[len(previousLayer.NeuralNodes)-1].UUID, currentNode.UUID, gradientDecentNetwork.NeuralConnections)

				for neuralConnectionIndex, neuralConnectionElement := range gradientDecentNetwork.NeuralConnections {

					if neuralConnectionElement.UUID == neuralConnection.UUID {

						gradientDecentNetwork.NeuralConnections[neuralConnectionIndex].Weight = gradientDecentNetwork.NeuralConnections[neuralConnectionIndex].Weight + (0.1 * currentNode.ErrorDelta)

					}
				}
			}

		}
	}

}

func (gradientDecentNetwork *GradientDescentNetwork) processWeightDeltas(newDeltas [][]float64) {

	fmt.Println("NEWDELTASTOSAVE:", newDeltas)
	for networkLayerIndex, networkLayerElement := range gradientDecentNetwork.NeuralLayers {
		usedIndex := 0

		for _, nodeElement := range networkLayerElement.NeuralNodes {


			for neuralConnectionIndex, neuralConnectElement := range gradientDecentNetwork.NeuralConnections {
				if neuralConnectElement.FromNodeUUID == nodeElement.UUID {
					fmt.Print("OLD-W: ", gradientDecentNetwork.NeuralConnections[neuralConnectionIndex].Weight)
					gradientDecentNetwork.NeuralConnections[neuralConnectionIndex].Weight = gradientDecentNetwork.NeuralConnections[neuralConnectionIndex].Weight +
						newDeltas[networkLayerIndex][usedIndex]

					fmt.Println("       NEW-W:", gradientDecentNetwork.NeuralConnections[neuralConnectionIndex].Weight)
					usedIndex = usedIndex + 1
				}

			}

		}

	}

}

func (gradientDecentNetwork *GradientDescentNetwork) SaveNetwork() {

	networkJson, _ := json.MarshalIndent(&gradientDecentNetwork, "", "    ")

	err := ioutil.WriteFile("output.json", networkJson, 0644)

	if err != nil {
		fmt.Println(err)
	}
}

func (gradientDecentNetwork *GradientDescentNetwork) LoadNetwork() GradientDescentNetwork {

	var gdN GradientDescentNetwork
	configFile, err := os.Open("StartPoint.json")

	defer configFile.Close()

	if err != nil {
		fmt.Println(err.Error())
	}

	jsonParser := json.NewDecoder(configFile)
	_ = jsonParser.Decode(&gdN)

	return gdN

}
