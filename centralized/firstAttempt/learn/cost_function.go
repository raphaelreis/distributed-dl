package learn

import "github.com/raphaelreis/distributed-dl/centralized/firstAttempt/network"

type CostFunction interface {
	CalculateDelta(layer *network.Layer, actualOutputs, idealOutputs []float64) []float64
}

// -------

var QuadraticCost CostFunction = newQuadraticCost()

type quadraticCost struct{}

func newQuadraticCost() CostFunction {
	q := quadraticCost{}
	return CostFunction(&q)
}

func (q *quadraticCost) CalculateDelta(layer *network.Layer, actualOutputs, idealOutputs []float64) []float64 {
	r := make([]float64, len(layer.Neurons))
	for i, neuron := range layer.Neurons {
		r[i] = (actualOutputs[i] - idealOutputs[i]) * neuron.CalculateOutputDelta()
	}
	return r
}

//// -------

var CrossEntropy CostFunction = newCrossEntropy()

type crossEntropy struct{}

func newCrossEntropy() CostFunction {
	c := crossEntropy{}
	return CostFunction(&c)
}

// TODO Optimize this function easy
func (c *crossEntropy) CalculateDelta(layer *network.Layer, actualOutputs, idealOutputs []float64) []float64 {
	r := make([]float64, len(layer.Neurons))
	for i := range layer.Neurons {
		r[i] = actualOutputs[i] - idealOutputs[i]
	}
	return r
}
