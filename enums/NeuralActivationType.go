package enums

type NeuralActivationType int

const (
	NeuralActivationSigmoid NeuralActivationType = 0
	NeuralActivationRelu    NeuralActivationType = 1
	NeuralActivationNone    NeuralActivationType = 99
)
