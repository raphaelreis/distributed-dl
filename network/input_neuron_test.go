package network

import (
	"reflect"
	"testing"
)

func TestNewInputNeuron(t *testing.T) {
	tests := []struct {
		name string
		want *InputNeuron
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewInputNeuron(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewInputNeuron() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestInputNeuron_ConnectTo(t *testing.T) {
	type fields struct {
		OutSynapses []*Synapse
		Input       float64
	}
	type args struct {
		layer Layer
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &InputNeuron{
				OutSynapses: tt.fields.OutSynapses,
				Input:       tt.fields.Input,
			}
			e.ConnectTo(tt.args.layer)
		})
	}
}

func TestInputNeuron_CreateSynapseTo(t *testing.T) {
	type fields struct {
		OutSynapses []*Synapse
		Input       float64
	}
	type args struct {
		nTo    *Neuron
		weight float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &InputNeuron{
				OutSynapses: tt.fields.OutSynapses,
				Input:       tt.fields.Input,
			}
			e.CreateSynapseTo(tt.args.nTo, tt.args.weight)
		})
	}
}

func TestInputNeuron_Trigger(t *testing.T) {
	type fields struct {
		OutSynapses []*Synapse
		Input       float64
	}
	tests := []struct {
		name   string
		fields fields
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &InputNeuron{
				OutSynapses: tt.fields.OutSynapses,
				Input:       tt.fields.Input,
			}
			e.Trigger()
		})
	}
}
