package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
)

// Generation represents a single fuzzing generation step.
type Generation struct {
	Generation         int     `json:"generation"`
	Time               float64 `json:"time"`
	TotalTransactions  int     `json:"total_transactions"`
	UniqueTransactions int     `json:"unique_transactions"`
	CodeCoverage       float64 `json:"code_coverage"`
	BranchCoverage     float64 `json:"branch_coverage"`
}

// ContractData holds the data for a single contract inside a JSON file.
type ContractData struct {
	Generations       []Generation `json:"generations"`
	ExecutionTime     float64      `json:"execution_time,omitempty"`
	MemoryConsumption float64      `json:"memory_consumption,omitempty"`
	Seed              float64      `json:"seed,omitempty"`
	Tag               string       `json:"tag,omitempty"`
	Algorithm         string       `json:"algorithm,omitempty"`
}

// Record is a flattened row of data (one per generation).
type Record struct {
	Filename           string
	RelativePath       string
	Folder             string
	Contract           string
	Generation         int
	TimeElapsed        float64
	GenerationTime     float64
	TotalTransactions  int
	UniqueTransactions int
	CodeCoverage       float64
	BranchCoverage     float64
	TotalExecutionTime float64
	MemoryConsumption  float64
	Seed               float64
	Tag                string
	Algorithm          string
}

// SmoothRecord includes a category field to distinguish global, small, and large.
type SmoothRecord struct {
	Algorithm    string
	TimeElapsed  float64
	CoverageType string // "code" or "branch"
	Category     string // "global", "small", "large"
	Mean         float64
	Std          float64
	Min          float64
	Max          float64
	SampleSize   int
}

func main() {
	// Command line arguments
	folderPath := flag.String("folder", "results", "Root folder containing JSON files")
	outputDir := flag.String("output", "conference_results_with_std", "Directory to write output files")
	timeInterval := flag.Float64("interval", 2.0, "Time interval for smoothing (seconds)")
	maxTime := flag.Float64("maxtime", 600.0, "Maximum time to consider (seconds)")
	threshold := flag.Int("threshold", 3632, "Transaction count threshold to separate small/large contracts")
	flag.Parse()

	// Ensure output directory exists
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output directory: %v\n", err)
		os.Exit(1)
	}

	// Step 1: Find all JSON files recursively
	jsonFiles, err := findJSONFiles(*folderPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error walking directory: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Found %d JSON files in %s\n", len(jsonFiles), *folderPath)

	// Step 2: Parse all JSON files and collect records
	records := []Record{}
	for _, filePath := range jsonFiles {
		fileRecords, err := parseJSONFile(filePath, *folderPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing %s: %v\n", filePath, err)
			continue
		}
		records = append(records, fileRecords...)
	}
	fmt.Printf("Parsed %d generation records\n", len(records))

	if len(records) == 0 {
		fmt.Println("No data found. Exiting.")
		return
	}

	// Step 3: Compute smoothed data for global, small, and large categories
	smoothRecords := computeSmoothedData(records, *timeInterval, *maxTime, *threshold)

	// Step 4: Write the aggregated CSV
	aggCSVPath := filepath.Join(*outputDir, "aggregated_smooth_data.csv")
	if err := writeAggregatedCSV(aggCSVPath, smoothRecords); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing aggregated CSV: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Aggregated data written to %s\n", aggCSVPath)

	fmt.Println("Go processing completed.")
}

// findJSONFiles recursively returns a list of all .json files under root.
func findJSONFiles(root string) ([]string, error) {
	var files []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && filepath.Ext(path) == ".json" {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

// parseJSONFile reads a JSON file, extracts all contracts, and returns a slice of Records.
func parseJSONFile(filePath, rootFolder string) ([]Record, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	var contracts map[string]ContractData
	if err := json.Unmarshal(data, &contracts); err != nil {
		return nil, err
	}

	absPath, _ := filepath.Abs(filePath)
	relPath, _ := filepath.Rel(rootFolder, absPath)
	folder := filepath.Dir(relPath)
	filename := filepath.Base(filePath)

	records := []Record{}
	for contractName, contractData := range contracts {
		algorithm := contractData.Algorithm
		if algorithm == "" {
			algorithm = "unknown"
		}

		cumulativeTime := 0.0
		for _, gen := range contractData.Generations {
			cumulativeTime += gen.Time
			records = append(records, Record{
				Filename:           filename,
				RelativePath:       relPath,
				Folder:             folder,
				Contract:           contractName,
				Generation:         gen.Generation,
				TimeElapsed:        cumulativeTime,
				GenerationTime:     gen.Time,
				TotalTransactions:  gen.TotalTransactions,
				UniqueTransactions: gen.UniqueTransactions,
				CodeCoverage:       gen.CodeCoverage,
				BranchCoverage:     gen.BranchCoverage,
				TotalExecutionTime: contractData.ExecutionTime,
				MemoryConsumption:  contractData.MemoryConsumption,
				Seed:               contractData.Seed,
				Tag:                contractData.Tag,
				Algorithm:          algorithm,
			})
		}
	}
	return records, nil
}

type stepEntry struct {
	time  float64
	value float64
}

// enforceNonDecreasing ensures values in step entries are non-decreasing over time.
// It modifies the slice in place.
func enforceNonDecreasing(entries []stepEntry) {
	if len(entries) == 0 {
		return
	}
	maxVal := entries[0].value
	for i := 1; i < len(entries); i++ {
		if entries[i].value < maxVal {
			entries[i].value = maxVal
		} else {
			maxVal = entries[i].value
		}
	}
}

// computeSmoothedData builds aggregated time series for global, small, and large contracts.
// It also ensures each contract's coverage is non-decreasing.
func computeSmoothedData(records []Record, interval, maxTime float64, threshold int) []SmoothRecord {
	// Group records by algorithm and contract
	byAlgo := make(map[string]map[string][]Record)
	for _, r := range records {
		if byAlgo[r.Algorithm] == nil {
			byAlgo[r.Algorithm] = make(map[string][]Record)
		}
		byAlgo[r.Algorithm][r.Contract] = append(byAlgo[r.Algorithm][r.Contract], r)
	}

	// Determine contract size classification per contract (based on max total_transactions)
	contractSize := make(map[string]map[string]bool) // algorithm -> contract -> true=large, false=small
	for algo, contracts := range byAlgo {
		contractSize[algo] = make(map[string]bool)
		for contract, recs := range contracts {
			maxTx := 0
			for _, r := range recs {
				if r.TotalTransactions > maxTx {
					maxTx = r.TotalTransactions
				}
			}
			contractSize[algo][contract] = maxTx > threshold
		}
	}

	timePoints := make([]float64, 0)
	for t := 0.0; t <= maxTime+1e-9; t += interval {
		timePoints = append(timePoints, t)
	}

	smoothRecords := []SmoothRecord{}

	// For each algorithm
	for algo, contracts := range byAlgo {
		// Build step functions for each contract (code and branch)
		codeSteps := make(map[string][]stepEntry)
		branchSteps := make(map[string][]stepEntry)

		for contract, recs := range contracts {
			// Sort records by time
			sort.Slice(recs, func(i, j int) bool {
				return recs[i].TimeElapsed < recs[j].TimeElapsed
			})
			code := make([]stepEntry, 0, len(recs))
			branch := make([]stepEntry, 0, len(recs))
			for _, r := range recs {
				code = append(code, stepEntry{time: r.TimeElapsed, value: r.CodeCoverage})
				branch = append(branch, stepEntry{time: r.TimeElapsed, value: r.BranchCoverage})
			}
			// Enforce monotonicity
			enforceNonDecreasing(code)
			enforceNonDecreasing(branch)

			codeSteps[contract] = code
			branchSteps[contract] = branch
		}

		// For each coverage type
		for _, coverageType := range []string{"code", "branch"} {
			var steps map[string][]stepEntry
			if coverageType == "code" {
				steps = codeSteps
			} else {
				steps = branchSteps
			}

			// For each time point
			for _, tp := range timePoints {
				// Collect values for global, small, large
				globalVals := []float64{}
				smallVals := []float64{}
				largeVals := []float64{}

				for contract, step := range steps {
					// Binary search for latest value at or before tp
					idx := sort.Search(len(step), func(i int) bool {
						return step[i].time > tp
					})
					if idx > 0 {
						val := step[idx-1].value
						globalVals = append(globalVals, val)
						if contractSize[algo][contract] {
							largeVals = append(largeVals, val)
						} else {
							smallVals = append(smallVals, val)
						}
					}
				}

				// Helper to add a record
				addRecord := func(category string, vals []float64) {
					if len(vals) == 0 {
						return
					}
					mean, std, minVal, maxVal := stats(vals)
					smoothRecords = append(smoothRecords, SmoothRecord{
						Algorithm:    algo,
						TimeElapsed:  tp,
						CoverageType: coverageType,
						Category:     category,
						Mean:         mean,
						Std:          std,
						Min:          minVal,
						Max:          maxVal,
						SampleSize:   len(vals),
					})
				}

				addRecord("global", globalVals)
				addRecord("small", smallVals)
				addRecord("large", largeVals)
			}
		}
	}

	return smoothRecords
}

// stats returns mean, standard deviation, min, max of a slice of float64.
func stats(vals []float64) (mean, std, minVal, maxVal float64) {
	if len(vals) == 0 {
		return 0, 0, 0, 0
	}
	minVal = vals[0]
	maxVal = vals[0]
	sum := 0.0
	for _, v := range vals {
		sum += v
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	mean = sum / float64(len(vals))

	// Standard deviation
	var sumSq float64
	for _, v := range vals {
		diff := v - mean
		sumSq += diff * diff
	}
	std = math.Sqrt(sumSq / float64(len(vals)))
	return
}

// writeAggregatedCSV writes the smoothed records to a single CSV.
func writeAggregatedCSV(filename string, records []SmoothRecord) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{"algorithm", "time_elapsed", "coverage_type", "category", "mean", "std", "min", "max", "sample_size"}
	if err := writer.Write(header); err != nil {
		return err
	}

	for _, r := range records {
		row := []string{
			r.Algorithm,
			strconv.FormatFloat(r.TimeElapsed, 'f', -1, 64),
			r.CoverageType,
			r.Category,
			strconv.FormatFloat(r.Mean, 'f', -1, 64),
			strconv.FormatFloat(r.Std, 'f', -1, 64),
			strconv.FormatFloat(r.Min, 'f', -1, 64),
			strconv.FormatFloat(r.Max, 'f', -1, 64),
			strconv.Itoa(r.SampleSize),
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}
	return nil
}