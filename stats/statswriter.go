package stats

import (
	"os"

	"github.com/patrikeh/go-deep"
)

// StatWriter write formatted neural net metrics to csv file
type StatWriter struct {
	folder string
}

// NewStatsWriter creates a StatsWriter
func NewStatsWriter(folder string) *StatWriter {
	return &StatWriter{folder}
}

// Init initialize the stat writer
func (w *StatWriter) Init(master *deep.Neural, workers []*deep.Neural) {
	os.Mkdir(w.folder, os.ModePerm)

}
