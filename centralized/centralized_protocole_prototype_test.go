package centralized

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

func TestCentralizedProtocole(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	train, err := load("/Users/Raphael/go/src/github.com/raphaelreis/distributed-dl/data/mnist_dataset/normalized/mnist_train_norm.csv")
	// train, err := load("/Users/Raphael/go/src/github.com/ldsec/dpnn/data/mnist_data/mnist_train.csv")
	if err != nil {
		panic(err)
	}
	test, err := load("/Users/Raphael/go/src/github.com/raphaelreis/distributed-dl/data/mnist_dataset/normalized/mnist_test_norm.csv")
	// test, err := load("/Users/Raphael/go/src/github.com/ldsec/dpnn/data/mnist_data/mnist_test.csv")
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

	//Paremeters
	epochs := 10

	net := deep.NewNeural(&deep.Config{
		Inputs:     len(train[0].Input),
		Layout:     []int{32, 10},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeMultiClass,
		Weight:     deep.NewNormal(0.6, 0.1), // slight positive bias helps ReLU
		Bias:       true,
	})

	// trainer := training.NewTrainer(training.NewSGD(0.01, 0.5, 1e-6, true), 1)
	trainer := training.NewBatchTrainer(training.NewSGD(0.01, 0.5, 1e-6, true), 1, 200, 8)

	fmt.Printf("training: %d, val: %d, test: %d\n", len(train), len(test), len(test))
	trainer.Train(net, train, test, epochs)

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
