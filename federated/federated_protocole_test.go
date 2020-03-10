package federated

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
)

func TestFederatedProtocole(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	train, err := load("/Users/Raphael/go/src/github.com/ldsec/dpnn/data/mnist_data/mnist_train.csv")
	if err != nil {
		panic(err)
	}
	test, err := load("/Users/Raphael/go/src/github.com/ldsec/dpnn/data/mnist_data/mnist_test.csv")
	if err != nil {
		panic(err)
	}

	for i := range train {
		for j := range train[i].Input {
			train[i].Input[j] = train[i].Input[j] / 255
		}
	}
	for i := range test {
		for j := range test[i].Input {
			test[i].Input[j] = test[i].Input[j] / 255
		}
	}
	test.Shuffle()
	train.Shuffle()

	// Constants
	numUsers := 5
	epochs := 10

	globalNetwork := deep.NewNeural(&deep.Config{
		Inputs:     len(train[0].Input),
		Layout:     []int{50, 10},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeMultiClass,
		Weight:     deep.NewNormal(0.6, 0.1), // slight positive bias helps ReLU
		Bias:       true,
	})

	trainer := training.NewBatchTrainer(training.NewSGD(0.01, 0.5, 0.999, true), 1, 200, 8)

	fmt.Printf("training: %d, val: %d, test: %d\n", len(train), len(test), len(test))

	// Iterate over epochs
	for e := 0; e < epochs; e++ {
		var w_locals []float64
		var loss_locals []float64
		// Iterate over the number of users
		for j := 0; j < numUsers; j++ {

		}
	}

	trainer.Train(globalNetwork, train, test, 500)
}

func load(path string) (training.Examples, error) {
	f, err := os.Open(path)
	defer f.Close()
	if err != nil {
		return nil, err
	}
	r := csv.NewReader(bufio.NewReader(f))

	var examples training.Examples
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		examples = append(examples, toExample(record))
	}

	return examples, nil
}

func toExample(in []string) training.Example {
	res, err := strconv.ParseFloat(in[0], 64)
	if err != nil {
		panic(err)
	}
	resEncoded := onehot(10, res)
	var features []float64
	for i := 1; i < len(in); i++ {
		res, err := strconv.ParseFloat(in[i], 64)
		if err != nil {
			panic(err)
		}
		features = append(features, res)
	}

	return training.Example{
		Response: resEncoded,
		Input:    features,
	}
}

func onehot(classes int, val float64) []float64 {
	res := make([]float64, classes)
	res[int(val)] = 1
	return res
}
