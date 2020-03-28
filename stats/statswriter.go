package stats

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
)

// StatWriter write formatted neural net metrics to csv file
type StatWriter struct {
	folder     string
	masterFile *os.File
	files      []*os.File
}

// NewStatsWriter creates a StatsWriter
func NewStatsWriter(folder string, master *deep.Neural, workers []*deep.Neural) *StatWriter {
	return &StatWriter{folder, new(os.File), make([]*os.File, len(workers))}
}

// Init initialize the stat writer
func (w *StatWriter) Init() {
	os.Mkdir(w.folder, os.ModePerm)
	_, err := os.Create(w.folder + "/master.csv")
	if err != nil {
		log.Fatal(err)
	}
	for i := 0; i < len(w.files); i++ {
		name := w.folder + "/worker" + strconv.Itoa(i) + ".csv"
		fmt.Println(name)
		_, err := os.Create(name)
		if err != nil {
			log.Fatal(err)
		}
	}
}

// WriteProgress write the current progress of the training
func (w *StatWriter) WriteProgress(master *deep.Neural, workers []*deep.Neural,
	validation training.Examples, elapsed time.Duration, iteration int) {

}

func accuracy(n *deep.Neural, validation training.Examples) float64 {
	correct := 0
	for _, e := range validation {
		est := n.Predict(e.Input)
		if deep.ArgMax(e.Response) == deep.ArgMax(est) {
			correct++
		}
	}
	return float64(correct) / float64(len(validation))
}

func crossValidate(n *deep.Neural, validation training.Examples) float64 {
	predictions, responses := make([][]float64, len(validation)), make([][]float64, len(validation))
	for i := 0; i < len(validation); i++ {
		predictions[i] = n.Predict(validation[i].Input)
		responses[i] = validation[i].Response
	}

	return deep.GetLoss(n.Config.Loss).F(predictions, responses)
}
