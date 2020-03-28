package stats

import (
	"os"
	"strconv"
	"testing"

	"github.com/patrikeh/go-deep"
)

func TestStatWriter_Init(t *testing.T) {

	defaultConfg := deep.Config{
		Inputs:     1,
		Layout:     []int{1},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeMultiClass,
		Weight:     deep.NewUniform(0, 0), // slight positive bias helps ReLU
		Bias:       false,
	}

	type fields struct {
		folder string
	}
	type args struct {
		master  *deep.Neural
		workers []*deep.Neural
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{
			name: "Create folder correctly",
			fields: fields{
				folder: "generatedStats",
			},
			args: args{
				master:  deep.NewNeural(&defaultConfg),
				workers: make([]*deep.Neural, 2),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := NewStatsWriter(tt.fields.folder, tt.args.master, tt.args.workers)
			w.Init()
			if _, err := os.Stat(tt.fields.folder); os.IsNotExist(err) {
				t.Error("Folder not created")
			}
			if _, err := os.Stat(tt.fields.folder + "/master.csv"); os.IsNotExist(err) {
				t.Error("master file does not exists")
			}
			for i := 0; i < len(tt.args.workers); i++ {
				if _, err := os.Stat(tt.fields.folder + "/worker" + strconv.Itoa(i) + ".csv"); os.IsNotExist(err) {
					t.Error("worker" + strconv.Itoa(i) + ".csv does not exists")
				}
			}
		})
	}
}
