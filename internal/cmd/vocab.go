package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"flag"
	"fmt"
	"go/format"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
)

const (
	packageName = "codec"
)

type config struct {
	url      string
	mapName  string
	filename string
}

func main() {
	encoding := flag.String("encoding", "", "encoding format. (e.g. cl100k_base)")
	flag.Parse()

	if encoding == nil {
		flag.PrintDefaults()
		os.Exit(1)
	}

	cfg := getConfig(*encoding)

	buf := new(bytes.Buffer)
	generatePreamble(buf, *encoding)
	generateVocabulary(buf, cfg.mapName, cfg.url)

	src, err := format.Source(buf.Bytes())
	if err != nil {
		log.Fatalf("error preparing source: %v", err)
	}

	if err := os.WriteFile(cfg.filename, src, 0o644); err != nil {
		log.Fatalf("error writing file: %v", err)
	}
}

func generatePreamble(w io.Writer, encoding string) {
	fmt.Fprintf(w, "// Code generated by internal/cmd/vocab.go. DO NOT EDIT.\n\n")
	fmt.Fprintf(w, "//go:generate go run ../internal/cmd/vocab.go -encoding %s\n\n", encoding)
	fmt.Fprintf(w, "package %s\n", packageName)
}

func generateVocabulary(w io.Writer, mapName string, uri string) {
	var respBody io.Reader
	if strings.HasPrefix(uri, "file:") {
		filePath := strings.TrimPrefix(uri, "file:")
		file, err := os.Open(filePath)
		if err != nil {
			log.Fatalf("error opening file: %v", err)
		}
		defer file.Close()
		respBody = file
	} else {
		respBody = nil // will be set by http.Get
		resp, err := http.Get(uri)
		if err != nil {
			log.Fatalf("error fetching file: %v", err)
		}
		respBody = resp.Body
		defer resp.Body.Close()
	}

	fmt.Fprintf(w, "import \"sync\"\n")
	fmt.Fprintf(w, "var (\n")
	fmt.Fprintf(w, "%v vocab\n", mapName)
	fmt.Fprintf(w, "%vOnce sync.Once\n", mapName)
	fmt.Fprintf(w, ")\n")
	fmt.Fprintf(w, "func %sInit() {\n", mapName)
	fmt.Fprintf(w, "%s = vocab{\n", mapName)

	scanner := bufio.NewScanner(respBody)
	first := true
	for scanner.Scan() {
		line := scanner.Text()
		if first && strings.Contains(line, "#version") {
			first = false
			continue // skip the version line
		}

		wordInput, idInput, ok := strings.Cut(line, " ")
		if !ok {
			log.Fatalf("invalid line: %q", line)
		}

		word, err := base64.StdEncoding.DecodeString(wordInput)
		if err != nil {
			log.Fatalf("invalid word: %q", wordInput)
		}

		id, err := strconv.ParseUint(idInput, 10, 0)
		if err != nil {
			log.Fatalf("invalid id: %q", idInput)
		}

		fmt.Fprintf(w, "%q: %d,\n", word, id)
	}

	fmt.Fprintf(w, "}\n}\n")
}

func getConfig(encoding string) config {
	switch encoding {
	case "o200k_base":
		return config{
			mapName:  "o200kBaseVocab",
			url:      "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
			filename: "o200k_base_vocab.go",
		}
	case "cl100k_base":
		return config{
			mapName:  "cl100kBaseVocab",
			url:      "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
			filename: "cl100k_base_vocab.go",
		}
	case "r50k_base":
		return config{
			mapName:  "r50kBaseVocab",
			url:      "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
			filename: "r50k_base_vocab.go",
		}
	case "p50k_base":
		return config{
			mapName:  "p50kBaseVocab",
			url:      "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
			filename: "p50k_base_vocab.go",
		}
	case "llama":
		return config{
			mapName:  "llamaBaseVocab",
			url:      "https://raw.githubusercontent.com/meta-llama/llama-models/refs/heads/main/models/llama3/tokenizer.model",
			filename: "llama_base_vocab.go",
		}
	case "gpt2":
		return config{
			mapName:  "gpt2BaseVocab",
			url:      "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
			filename: "gpt2_base_vocab.go",
		}
	case "anthropic":
		return config{
			mapName: "anthropicVocab",
			// https://huggingface.co/Xenova/claude-tokenizer/blob/main/tokenizer.json
			url:      "file:internal/anthropic/anthropic.tiktoken",
			filename: "anthropic_base_vocab.go",
		}
	// case "sentencepiece":
	// 	return config{
	// 		mapName:  "sentencePieceVocab",
	// 		url:      "https://github.com/google/gemma_pytorch/raw/refs/heads/main/tokenizer/tokenizer.model",
	// 		filename: "sentencepiece_vocab.go",
	// 	}
	// case "bert":
	// 	return config{
	// 		mapName:  "bertVocab",
	// 		url:      "https://raw.githubusercontent.com/google-research/bert/master/vocab.txt",
	// 		filename: "bert_vocab.go",
	// 	}
	default:
		log.Fatal("config not found")
		return config{}
	}
}
