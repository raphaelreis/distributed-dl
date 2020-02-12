package trainingdata

import (
	"fmt"

	"github.com/petar/GoMNIST"
)

type MnistData struct {
	trainingData *GoMNIST.Set
	testData     *GoMNIST.Set
}

func NewMnistData() MnistData {
	train, test, err := GoMNIST.Load("data/mnist_dataset")
	if err != nil {
		panic(err)
	}

	return MnistData{
		trainingData: train,
		testData:     test,
	}
}

func (m *MnistData) MakeTrainingData(sample float64) []TrainingData {
	fmt.Printf("Number of train samples: %v\n", int(float64(m.trainingData.Count())*sample))
	return m.marshallData(m.trainingData, sample)
}

func (m *MnistData) MakeTestData(sample float64) []TrainingData {
	fmt.Printf("Number of test samples: %v\n", int(float64(m.testData.Count())*sample))
	return m.marshallData(m.testData, sample)
}

func (m *MnistData) marshallData(dataset *GoMNIST.Set, sample float64) []TrainingData {
	result := make([]TrainingData, 0, dataset.Count())
	sampleCount := int(float64(dataset.Count()) * sample)
	for i := 0; i < sampleCount; i++ {
		image, label := dataset.Get(i)

		vectorizedImage := make([]float64, 0, 784)
		bounds := image.Bounds()
		for x := 0; x < bounds.Max.X; x++ {
			for y := 0; y < bounds.Max.Y; y++ {
				r, _, _, _ := image.At(x, y).RGBA()
				vectorizedImage = append(vectorizedImage, float64(r)/100000)
			}
		}
		t := TrainingData{
			TrainingInput:  vectorizedImage,
			DesiredOutputs: m.vectorizeOutput(uint8(label)),
		}
		result = append(result, t)
	}

	return result
}

func (m *MnistData) vectorizeOutput(n uint8) []float64 {
	res := make([]float64, 10)
	res[n] = 1.0
	return res
}
