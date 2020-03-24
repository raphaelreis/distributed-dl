package main

import (
	"fmt"

	"github.com/raphaelreis/distributed-dl/centralized/firstAttempt/learn"
	"github.com/raphaelreis/distributed-dl/centralized/firstAttempt/network"
	"github.com/raphaelreis/distributed-dl/trainingdata"
)

func main() {

	sample := 1.
	mnistData := trainingdata.NewMnistData()
	trainingData := mnistData.MakeTrainingData(sample)
	testData := mnistData.MakeTestData(sample)

	batch := [2]int{128}
	neurons := []int{128}
	activation := [2]string{"sigmoid"}
	regularizer := [3]string{"l1", "l2", "none"}
	costFunction := learn.CrossEntropy
	learningRate := 0.1
	lambda := 5.0
	reportResults := false
	epochs := 10

	for _, b := range batch {
		for _, n := range neurons {
			for _, act := range activation {
				for _, reg := range regularizer {
					fmt.Printf("Model parameters:\nneurons: %v | batchSize: %v | activation: %v | regularizer: %v\n", n, b, act, reg)
					layers := []int{n, 10}
					nn := network.NewNetwork(784, layers, act)
					parsedRegularization := parseRegularization(reg)
					nt := learn.NewNetworkTrainer(nn, trainingData, costFunction, parsedRegularization, learningRate, lambda, reportResults)
					outFileName := fmt.Sprintf("./data/rawneuroninput/batch%vneurons%v_%v_reg_%v.csv", b, n, act, reg)
					nt.TrainByGradientDescent(epochs, b, testData, outFileName)
				}
			}
		}
	}
}

func parseRegularization(e string) learn.RegularizationFunction {
	switch e {
	case "none":
		return learn.NoRegularization
	case "l1":
		return learn.L1Regularization
	case "l2":
		return learn.L2Regularization
	}

	return learn.L2Regularization
}
