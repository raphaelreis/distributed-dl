package federated_test

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
)

func TestFederatedProtocole(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	train, err := load("/Users/Raphael/go/src/github.com/ldsec/dpnn/data/mnist_data/normalized/mnist_train_norm.csv")
	if err != nil {
		panic(err)
	}
	test, err := load("/Users/Raphael/go/src/github.com/ldsec/dpnn/data/mnist_data/normalized/mnist_test_norm.csv")
	if err != nil {
		panic(err)
	}

	// for i := range train {
	// 	for j := range train[i].Input {
	// 		train[i].Input[j] = train[i].Input[j] / 255
	// 	}
	// }
	// for i := range test {
	// 	for j := range test[i].Input {
	// 		test[i].Input[j] = test[i].Input[j] / 255
	// 	}
	// }
	test.Shuffle()
	train.Shuffle()

	// Constants
	numUsers := 5
	epochs := 1

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

	// Prepare networks and data
	// workCh := make(chan Example, numUsers)
	nets := make([]*deep.Neural, numUsers)
	for i := 0; i < numUsers; i++ {
		nets[i] = deep.NewNeural(globalNetwork.Config)
	}
	nonOverlappingData := train.SplitSize(numUsers)
	var globalWeights [][][]float64

	// Iterate over epochs
	for e := 0; e < epochs; e++ {

		if e != 0 {
			for i := 0; i < numUsers; i++ {
				nets[i].ApplyWeights(globalWeights)
			}
		}

		var wg sync.WaitGroup
		// Iterate over the number of users
		wg.Add(len(nonOverlappingData))
		for i, b := range nonOverlappingData {
			go func(batch []training.Example) {

				// trainer = training.NewBatchTrainer(training.NewAdam(0.02, 0.9, 0.999, 1e-8), 1, 200, 8)
				trainer.Train(nets[i], batch, test, 1)

				wg.Done()
			}(b)
		}
		wg.Wait()

		globalWeights = federatedWeights(nets, numUsers)
	}
}

func federatedWeights(nets []*deep.Neural, numUsers int) [][][]float64 {

	fedWeights := nets[0].Weights()
	for i, net := range nets {
		if i != 0 {
			fedWeights = add3d(fedWeights, net.Weights())
		}
	}
	nu := float64(numUsers)
	for i := range fedWeights {
		for j := range fedWeights[i] {
			for k := range fedWeights[i][j] {
				fedWeights[i][j][k] /= nu
			}
		}
	}

	return fedWeights
}

func add(a, b []float64) []float64 {

	sum := make([]float64, len(a))
	for i, a_ := range a {
		sum[i] = a_ + b[i]
	}
	return sum
}

func add2d(a, b [][]float64) [][]float64 {

	sum := make([][]float64, len(a))
	for i, a_ := range a {
		sum[i] = make([]float64, len(a_))
	}

	for i, a_ := range a {
		sum[i] = add(a_, b[i])
	}

	return sum
}

func add3d(a, b [][][]float64) [][][]float64 {
	sum := a
	for i := range a {
		sum[i] = add2d(sum[i], b[i])
	}
	return sum
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
