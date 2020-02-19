package utils

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
)

// ArrayToString converts a []float64 slice to a []string with
// a delimiter `delim`
func ArrayToString(values []float64) []string {
	valuesText := []string{}
	for _, val := range values {
		text := fmt.Sprintf("%f", val)
		valuesText = append(valuesText, text)
	}
	return valuesText
}

// MatrixToString converts a 2d slice into a string formatted
// with a `elem_delim` to seperate row values and `row_delim`
// to split rows
func MatrixToString(values [][]float64) [][]string {
	valuesText := [][]string{}

	for _, val := range values {
		stringArray := ArrayToString(val)
		valuesText = append(valuesText, stringArray)
	}
	return valuesText
}

// SaveMatrixCsv save as CSV file a [][]float64 slice
func SaveMatrixCsv(values [][]float64, filePath string) {
	s := MatrixToString(values)

	csvfile, err := os.Create(filePath)
	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}
	csvwriter := csv.NewWriter(csvfile)
	csvwriter.WriteAll(s)
	csvwriter.Flush()
	csvfile.Close()
}
