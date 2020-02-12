package trainingdata

import (
	"reflect"
	"testing"
)

func TestShuffleTrainingData(t *testing.T) {
	type args struct {
		td []TrainingData
	}
	tests := []struct {
		name string
		args args
		want []TrainingData
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ShuffleTrainingData(tt.args.td); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ShuffleTrainingData() = %v, want %v", got, tt.want)
			}
		})
	}
}
