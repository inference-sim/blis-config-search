// engine-check validates experiment-spec engine blocks with blis-vllm itself: the strict
// engineconfig.Decode (an unknown or misplaced key is an error at its line) and the vLLM v0.32
// rules pack (values the release refuses). scripts/run_experiment.py calls it once per spec with
// every distinct engine block the grids expand to, so a spec that names a knob vLLM does not take
// fails before any BLIS run.
//
// Input (stdin, JSON):
//
//	{"graph": "<catalog>/models/<m>/graph.yaml" | "", "hardware": "h100",
//	 "gpus_per_node": 8,
//	 "cases": [{"id": "...", "nodes": 1, "parallel": {...}, "engine": {...}}]}
//
// Output (stdout, JSON): {"graph_loaded": bool, "graph_error": "...",
//
//	"cases": [{"id": "...", "errors": [...], "warnings": [...]}]}
//
// in input order. The exit status is 0 whenever the input parsed; findings are data.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	blisschemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-schemas/spec/scenario"
	"github.com/inference-sim/blis-vllm/rules/v0_32"
	"gopkg.in/yaml.v3"
)

type input struct {
	Graph       string `json:"graph"`
	Hardware    string `json:"hardware"`
	GPUsPerNode int    `json:"gpus_per_node"`
	Cases       []struct {
		ID       string         `json:"id"`
		Nodes    int            `json:"nodes"`
		Parallel map[string]any `json:"parallel"`
		Engine   map[string]any `json:"engine"`
	} `json:"cases"`
}

type caseOut struct {
	ID       string   `json:"id"`
	Errors   []string `json:"errors"`
	Warnings []string `json:"warnings"`
}

type output struct {
	Version     string    `json:"rules_version"`
	GraphLoaded bool      `json:"graph_loaded"`
	GraphError  string    `json:"graph_error,omitempty"`
	Cases       []caseOut `json:"cases"`
}

func main() {
	var in input
	if err := json.NewDecoder(os.Stdin).Decode(&in); err != nil {
		fmt.Fprintf(os.Stderr, "engine-check: reading input: %v\n", err)
		os.Exit(2)
	}
	if in.GPUsPerNode == 0 {
		in.GPUsPerNode = 8
	}
	pack := v0_32.Pack()
	out := output{Version: pack.Version, Cases: []caseOut{}}

	// The model graph lets the rules that depend on the architecture fire (a DeepSeek-V4.1
	// cache_dtype, a DSA block size). A graph this schemas release cannot read is reported, and the
	// architecture-free checks still run.
	var graph *model.Graph
	if in.Graph != "" {
		g, err := blisschemas.LoadModelGraph(in.Graph)
		if err != nil {
			out.GraphError = err.Error()
		} else {
			graph, out.GraphLoaded = g, true
		}
	}

	for _, c := range in.Cases {
		out.Cases = append(out.Cases, check(pack, graph, in, c.ID, c.Nodes, c.Parallel, c.Engine))
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", " ")
	if err := enc.Encode(out); err != nil {
		os.Exit(2)
	}
}

func check(pack rules.EngineRules, graph *model.Graph, in input, id string, nodes int,
	parallel, engine map[string]any) caseOut {
	res := caseOut{ID: id, Errors: []string{}, Warnings: []string{}}
	if nodes < 1 {
		nodes = 1
	}
	// The deployment is decoded from YAML exactly as a scenario file would be, strictly, so the
	// engine block reaches blis-vllm as the same yaml node a real run's would.
	doc := map[string]any{
		"kind": "Deployment", "name": id,
		"pools": []any{map[string]any{
			"role": "colocated", "nodes": nodes, "parallel": parallel, "engine": engine,
		}},
	}
	raw, err := yaml.Marshal(doc)
	if err != nil {
		res.Errors = append(res.Errors, err.Error())
		return res
	}
	var d deployment.Deployment
	dec := yaml.NewDecoder(strings.NewReader(string(raw)))
	dec.KnownFields(true)
	if err := dec.Decode(&d); err != nil {
		res.Errors = append(res.Errors, err.Error())
		return res
	}
	collect := func(p *rules.Problems) {
		for _, pr := range p.All() {
			if pr.Severity == rules.SeverityError {
				res.Errors = append(res.Errors, pr.String())
			} else {
				res.Warnings = append(res.Warnings, pr.String())
			}
		}
	}
	// Field layer: strict engineconfig.Decode plus engineconfig.Validate.
	blockProblems := pack.CheckEngineBlock("pools[0]", d.Pools[0].Engine)
	collect(blockProblems)
	if !blockProblems.OK() {
		return res
	}
	sc := &scenario.Scenario{Kind: "Scenario", Name: id, EngineVersion: v0_32.Version,
		Cluster: scenario.Cluster{Hardware: in.Hardware, Nodes: nodes, GPUsPerNode: in.GPUsPerNode}}
	if graph != nil {
		sc.Model = graph.Name
	}
	collect(pack.Check(rules.Input{Scenario: sc, Deployment: &d, Model: graph}))
	return res
}
