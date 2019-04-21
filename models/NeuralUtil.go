package models

func FindNeuralConnection(fromUUID string, toUUID string, neuralConnections []NeuralConnection) NeuralConnection {

	var neuralConnection NeuralConnection

	for _, element := range neuralConnections {
		if element.FromNodeUUID == fromUUID {
			if element.ToNodeUUID == toUUID {
				return element
			}
		}
	}

	return neuralConnection

}

func FindReceivingConnections(receivingUUID string, neuralConnections []NeuralConnection) []NeuralConnection {

	var nerualConnections []NeuralConnection

	for _, element := range neuralConnections {

		if element.ToNodeUUID == receivingUUID {
			neuralConnections = append(neuralConnections, element)
		}
	}

	return nerualConnections
}

func FindSendingConnections(sendingUUID string, neuralConnections []NeuralConnection) []NeuralConnection {

	var nerualConnections []NeuralConnection

	for _, element := range neuralConnections {

		if element.FromNodeUUID == sendingUUID {
			neuralConnections = append(neuralConnections, element)
		}
	}

	return nerualConnections
}