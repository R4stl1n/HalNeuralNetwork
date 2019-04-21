package enums

type NeuralActivationType int

const (
	NeuralActivationSigmoid         NeuralActivationType = 0
	NeuralActivationRelu            NeuralActivationType = 1
	NeuralActivationSigmoidTransfer NeuralActivationType = 2
	NeuralActivationNone            NeuralActivationType = 99
)
