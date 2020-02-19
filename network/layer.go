package network

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(neurons int, activation string) *Layer {
	l := &Layer{}
	l.init(neurons, activation)
	return l
}

func (l *Layer) getNeuronsValues() []float64 {
	inputTracker := make([]float64, len(l.Neurons))
	for i := range l.Neurons {
		inputTracker = append(inputTracker, l.Neurons[i].getNeuronValue())
	}
	return inputTracker
}

func (l *Layer) init(neurons int, activation string) {
	n := make([]*Neuron, 0, neurons)

	for i := 0; i < neurons; i++ {
		neuron := new(Neuron)
		neuron.Activation = activation
		n = append(n, neuron)
	}

	l.Neurons = n
}

func (l *Layer) ConnectTo(layer *Layer) {
	for _, n := range l.Neurons {
		for _, toN := range layer.Neurons {
			n.CreateSynapseTo(toN, 0)
		}
	}
}

func (l *Layer) CalculateNewOutputs() {
	for i := range l.Neurons {
		l.Neurons[i].CalculateAndSendOutput()
	}
}
