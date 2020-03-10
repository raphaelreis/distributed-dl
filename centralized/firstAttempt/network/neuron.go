package network

import (
	"fmt"
	"math"
)

type Neuron struct {
	InSynapses  []*Synapse
	OutSynapses []*Synapse
	RawInput    float64
	Bias        float64
	Out         float64
	Activation  string
}

func (n *Neuron) getNeuronValue() float64 {
	return n.RawInput
}
func (n *Neuron) CreateSynapseTo(nTo *Neuron, weight float64) {
	NewSynapseFromTo(n, nTo, weight)
}

func (n *Neuron) CalculateWeightedInput() float64 {
	var sum float64

	for _, s := range n.InSynapses {
		sum += s.Out
	}

	sum += n.Bias

	return sum
}

func (n *Neuron) CalculateOutputDelta() float64 {
	z := n.CalculateWeightedInput()

	return n.activationFunctionPrime(z)
}

func (n *Neuron) CalculateOutput() float64 {
	n.RawInput = n.CalculateWeightedInput()

	return n.activationFunction(n.RawInput)
}

func (n *Neuron) CalculateAndSendOutput() {
	n.Out = n.CalculateOutput()

	for _, s := range n.OutSynapses {
		s.Trigger(n.Out)
	}
}

func (n *Neuron) activationFunctionPrime(z float64) float64 {
	if n.Activation == "sigmoid" { // Sigmoid
		return n.activationFunction(z) * (1 - n.activationFunction(z))
	} else if n.Activation == "relu" { // Relu
		if z > 0. {
			return 1.
		} else {
			return 0.
		}
	} else {
		panic(fmt.Sprintf("Activation function: %v", n.Activation))
	}
}

func (n *Neuron) activationFunction(z float64) float64 {
	if n.Activation == "sigmoid" { // Sigmoid
		return 1.0 / (1.0 + math.Exp(-z))
	} else if n.Activation == "relu" { // Relu
		return math.Max(0., z)
	} else {
		panic(fmt.Sprintf("Activation function: %v", n.Activation))
	}

}
