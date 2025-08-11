package tokenizer

// Package tokenizer provides functions for encoding and decoding text using
// different tokenization schemes.
//
// Encoding Formats
//
// The following encoding formats are supported:
// - Cl100kBase
// - R50kBase
// - P50kBase
// - P50kEdit
// - LLamaBase
// - GPT2Enc [hope to support]
//
// Alternatively you can request a tokenizer using OpenAI's model name, the
// following OpenAI models are supported:
// - O4Mini
// - O3
// - O3Mini
// - O1
// - O1Preview
// - O1Mini
// - GPT4.1
// - GPT4o
// - GPT4
// - GPT35Turbo
// - TextEmbeddingAda002
// - TextDavinci003
// - TextDavinci002
// - CodeDavinci002
// - CodeDavinci001
// - CodeCushman002
// - CodeCushman001
// - DavinciCodex
// - CushmanCodex
// - TextDavinci001
// - TextCurie001
// - TextBabbage001
// - TextAda001
// - Davinci
// - Curie
// - Babbage
// - Ada
// - TextSimilarityDavinci001
// - TextSimilarityCurie001
// - TextSimilarityBabbage001
// - TextSimilarityAda001
// - TextSearchDavinciDoc001
// - TextSearchCurieDoc001
// - TextSearchAdaDoc001
// - TextSearchBabbageDoc001
// - CodeSearchBabbageCode001
// - CodeSearchAdaCode001
// - TextDavinciEdit001
// - CodeDavinciEdit001
// - LLama3
//
// Usage Example
//
// Here is an example of how to encode a string using the `ForModel` function:
//
//	package main
//
//	import (
//		"fmt"
//		"github.com/awee-ai/go-tokenizer"
//	)
//
//	func main() {
//		enc, err := tokenizer.Get(tokenizer.Cl100kBase)
//		if err != nil {
//			panic("oh oh")
//		}
//
//		// this should print a list of token ids
//		ids, token, _ := enc.Encode("supercalifragilistic")
//		fmt.Println(ids)
//
//		// this should print the original string back
//		text, _ := enc.Decode(ids)
//		fmt.Println(text)
// }

import (
	"errors"
	"strings"

	"github.com/awee-ai/go-tokenizer/codec"
)

var (
	ErrModelNotSupported    = errors.New("model not supported")
	ErrEncodingNotSupported = errors.New("encoding not supported")
)

type Codec interface {
	GetName() string
	Count(string) (int, error)
	Encode(string) ([]uint, []string, error)
	Decode([]uint) (string, error)
}

type Model string

const (
	O1                       Model = "o1"
	O1Preview                Model = "o1-preview"
	O1Mini                   Model = "o1-mini"
	O3                       Model = "o3"
	O3Mini                   Model = "o3-mini"
	O4Mini                   Model = "o4-mini"
	GPT41                    Model = "gpt-4.1"
	GPT4o                    Model = "gpt-4o"
	GPT4                     Model = "gpt-4"
	GPT35Turbo               Model = "gpt-3.5-turbo"
	GPT35                    Model = "gpt-3.5"
	TextEmbeddingAda002      Model = "text-embedding-ada-002"
	TextDavinci003           Model = "text-davinci-003"
	TextDavinci002           Model = "text-davinci-002"
	CodeDavinci002           Model = "code-davinci-002"
	CodeDavinci001           Model = "code-davinci-001"
	CodeCushman002           Model = "code-cushman-002"
	CodeCushman001           Model = "code-cushman-001"
	DavinciCodex             Model = "davinci-codex"
	CushmanCodex             Model = "cushman-codex"
	TextDavinci001           Model = "text-davinci-001"
	TextCurie001             Model = "text-curie-001"
	TextBabbage001           Model = "text-babbage-001"
	TextAda001               Model = "text-ada-001"
	Davinci                  Model = "davinci"
	Curie                    Model = "curie"
	Babbage                  Model = "babbage"
	Ada                      Model = "ada"
	TextSimilarityDavinci001 Model = "text-similarity-davinci-001"
	TextSimilarityCurie001   Model = "text-similarity-curie-001"
	TextSimilarityBabbage001 Model = "text-similarity-babbage-001"
	TextSimilarityAda001     Model = "text-similarity-ada-001"
	TextSearchDavinciDoc001  Model = "text-search-davinci-doc-001"
	TextSearchCurieDoc001    Model = "text-search-curie-doc-001"
	TextSearchAdaDoc001      Model = "text-search-ada-doc-001"
	TextSearchBabbageDoc001  Model = "text-search-babbage-doc-001"
	CodeSearchBabbageCode001 Model = "code-search-babbage-code-001"
	CodeSearchAdaCode001     Model = "code-search-ada-code-001"
	TextDavinciEdit001       Model = "text-davinci-edit-001"
	CodeDavinciEdit001       Model = "code-davinci-edit-001"
)

type Encoding string

const (
	R50kBase        Encoding = "r50k_base"   // OpenAI GPT-2 base tokenizer (same as above)
	P50kBase        Encoding = "p50k_base"   // Codex tokenizer variant (GPT-style BPE, 50k)
	P50kEdit        Encoding = "p50k_edit"   // Used by OpenAI's edit models
	Cl100kBase      Encoding = "cl100k_base" // GPT-4/GPT-3.5 Turbo tokenizer (100k BPE)
	O200kBase       Encoding = "o200k_base"  // OpenAI 200k tokenizer (e.g. GPT-4o, o1, o3)
	OllamaLlamaBase Encoding = "llama"       // LLaMA3 tokenizer (BPE, 200k vocab, used by LLama3+ models)

	// r50k_base
	// expected: 91
	// actual: 79

	// p50k_base
	// expected: 91
	// actual: 78

	// p50k_edit
	// expected: 91
	// actual: 78

	// cl100k_base
	// expected: 91
	// actual: 80

	// o200k_base
	// expected: 91
	// actual: 78

	// llama
	// expected: 91
	// actual: 80

	AnthropicBase Encoding = "cl100k_base" // Anthropic tokenizer (Claude family, 100k vocab)
)

// DeepSeek family - custom tokenizer but GPT-2 style BPE, vocab >100k
var deepSeekModels = map[string]Encoding{
	"deepseek-r1":       R50kBase, // Fallback to GPT-2 style, vocab size will be different
	"deepseek-v3":       R50kBase,
	"deepseek-v2.5":     R50kBase,
	"deepseek-v2":       R50kBase,
	"deepseek-coder-v2": R50kBase,
	"deepseek-coder":    R50kBase,
	"deepseek-llm":      R50kBase,
	"deepcoder":         R50kBase,
	"deepscaler":        R50kBase,
}

var definitiveTokenizerFamilies = map[string]Encoding{
	"gpt-5": O200kBase,
	"o1-":   O200kBase,
	"o3-":   O200kBase,
	"o4-":   O200kBase,
	// chat
	"chatgpt-4o-":    O200kBase,
	"gpt-4.1-":       O200kBase,
	"gpt-4o-":        O200kBase,
	"gpt-4-":         Cl100kBase,
	"gpt-3.5-turbo-": Cl100kBase,
	"gpt-35-turbo-":  Cl100kBase,
	// fine-tuned
	"ft:gpt-4":         Cl100kBase,
	"ft:gpt-3.5-turbo": Cl100kBase,
	"ft:davinci-002":   Cl100kBase,
	"ft:babbage-002":   Cl100kBase,
}

// Llama family - complex because Llama 2 vs 3+ have different tokenizers
var llamaModels = map[string]Encoding{
	// Llama 3+ family uses tiktoken-style with ~200k vocab
	"llama3.1": OllamaLlamaBase,
	"llama3.2": OllamaLlamaBase,
	"llama3.3": OllamaLlamaBase,
	"llama3":   OllamaLlamaBase,
	"llama4":   OllamaLlamaBase,

	// Llama 2 family uses SentencePiece - fallback to GPT-2 style for now
	"llama2":            R50kBase, // MIGRATION: Should be SentencePiece
	"codellama":         R50kBase, // Based on Llama 2, MIGRATION: Should be SentencePiece
	"llama2-uncensored": R50kBase, // MIGRATION: Should be SentencePiece
	"llama2-chinese":    R50kBase, // MIGRATION: Should be SentencePiece
}

// Qwen family - custom tiktoken-compatible implementation
var qwenModels = map[string]Encoding{
	"qwen3":         R50kBase, // Custom but tiktoken-compatible
	"qwen2.5vl":     R50kBase,
	"qwen2.5":       R50kBase,
	"qwen2.5-coder": R50kBase,
	"qwen":          R50kBase,
	"qwen2":         R50kBase,
	"qwen2-math":    R50kBase,
	"qwq":           R50kBase,
	"codeqwen":      R50kBase,
}

var claudeModels = map[string]Encoding{
	"claude-2.0": Cl100kBase,
	"claude-2.1": Cl100kBase,

	// Cl100kBase
	// expected: 49
	// actual  : 41
	"claude-3-opus-":     AnthropicBase,
	"claude-3-sonnet-":   AnthropicBase,
	"claude-3-5-sonnet-": AnthropicBase,
	"claude-3-haiku-":    AnthropicBase,
	"claude-3-5-haiku-":  AnthropicBase,
	"claude-3-7-sonnet-": AnthropicBase,

	"claude-opus-4":   AnthropicBase,
	"claude-sonnet-4": AnthropicBase,

	//
	//
	// "claude-3-opus-":     R50kBase,
	// "claude-3-sonnet-":   R50kBase,
	// "claude-3-5-sonnet-": R50kBase,
	// "claude-3-haiku-":    R50kBase,
	// "claude-3-5-haiku-":  R50kBase,
	// "claude-3-7-sonnet-": R50kBase,
	//
	// "claude-opus-4":   R50kBase,
	// "claude-sonnet-4": R50kBase,
}

// Mistral family - mixed tokenizers (older=SentencePiece, newer=Tekken/tiktoken)
var mistralModels = map[string]Encoding{
	"mistral":          R50kBase, // MIGRATION: Older versions use SentencePiece
	"mistral-nemo":     R50kBase, // Uses Tekken (tiktoken-based)
	"mistral-small":    R50kBase, // MIGRATION: Older versions use SentencePiece
	"mistral-small3.1": R50kBase, // Uses Tekken
	"mistral-small3.2": R50kBase, // Uses Tekken
	"mistral-large":    R50kBase, // MIGRATION: Check version
	"mistral-openorca": R50kBase, // MIGRATION: Likely SentencePiece
	"mistrallite":      R50kBase, // MIGRATION: Likely SentencePiece
	"mathstral":        R50kBase, // MIGRATION: Likely SentencePiece
	"codestral":        R50kBase, // MIGRATION: Likely SentencePiece
	"devstral":         R50kBase, // Recent, likely Tekken
	"mixtral":          R50kBase, // MIGRATION: SentencePiece
}

// Gemma family - SentencePiece with BPE
var gemmaModels = map[string]Encoding{
	"gemma3n":   R50kBase, // MIGRATION: Should be SentencePiece
	"gemma3":    R50kBase, // MIGRATION: Should be SentencePiece
	"gemma2":    R50kBase, // MIGRATION: Should be SentencePiece
	"gemma":     R50kBase, // MIGRATION: Should be SentencePiece
	"codegemma": R50kBase, // MIGRATION: Should be SentencePiece
}

// Phi family - tokenizer changed between versions
var phiModels = map[string]Encoding{
	"phi3":                OllamaLlamaBase,
	"phi4":                R50kBase, // Uses tiktoken (100k vocab)
	"phi4-mini":           R50kBase, // Uses tiktoken
	"phi4-reasoning":      R50kBase, // Uses tiktoken
	"phi4-mini-reasoning": R50kBase, // Uses tiktoken
	"phi3.5":              R50kBase, // MIGRATION: Uses SentencePiece
	"phi":                 R50kBase, // Phi-2 uses CodeGen tokenizer
}

// Vision/Multimodal models - inherit from base model
var visionModels = map[string]Encoding{
	"llava":             R50kBase,        // MIGRATION: Depends on base (Llama 2 = SentencePiece)
	"llava-llama3":      OllamaLlamaBase, // Based on Llama 3
	"llava-phi3":        R50kBase,        // MIGRATION: Based on Phi-3 (SentencePiece)
	"minicpm-v":         R50kBase,        // MIGRATION: Likely SentencePiece
	"llama3.2-vision":   OllamaLlamaBase, // Based on Llama 3.2
	"bakllava":          R50kBase,        // MIGRATION: Based on Mistral (SentencePiece)
	"moondream":         R50kBase,        // Custom small model
	"granite3.2-vision": R50kBase,        // IBM custom
}

// Granite family - IBM models
var graniteModels = map[string]Encoding{
	"granite-code":      R50kBase, // IBM custom tokenizer
	"granite3.3":        R50kBase,
	"granite3.2":        R50kBase,
	"granite3.1-dense":  R50kBase,
	"granite3.1-moe":    R50kBase,
	"granite3-dense":    R50kBase,
	"granite3-moe":      R50kBase,
	"granite3-guardian": R50kBase,
	"granite-embedding": R50kBase,
}

// Small/Specialized models - mostly use smaller variants
var smallModels = map[string]Encoding{
	"smollm2":     R50kBase,
	"smollm":      R50kBase,
	"tinyllama":   R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"phi":         R50kBase, // Phi-2
	"tinydolphin": R50kBase, // MIGRATION: Based on TinyLlama (SentencePiece)
}

// Embedding models - typically simpler tokenizers
var embeddingModels = map[string]Encoding{
	"nomic-embed-text":        R50kBase, // MIGRATION: Likely custom/SentencePiece
	"mxbai-embed-large":       R50kBase,
	"bge-m3":                  R50kBase,
	"snowflake-arctic-embed":  R50kBase,
	"snowflake-arctic-embed2": R50kBase,
	"all-minilm":              R50kBase,
	"bge-large":               R50kBase,
	"paraphrase-multilingual": R50kBase,
}

// Models based on other models - inherit tokenizer
var derivedModels = map[string]Encoding{
	// Dolphin family - inherits from base model
	"dolphin3":        OllamaLlamaBase, // Based on Llama 3.1
	"dolphin-mixtral": R50kBase,        // MIGRATION: Based on Mixtral (SentencePiece)
	"dolphin-mistral": R50kBase,        // MIGRATION: Based on Mistral (SentencePiece)
	"dolphin-llama3":  OllamaLlamaBase, // Based on Llama 3
	"dolphincoder":    R50kBase,        // MIGRATION: Based on StarCoder (custom)
	"dolphin-phi":     R50kBase,        // MIGRATION: Based on Phi (CodeGen)

	// Hermes family
	"hermes3":              OllamaLlamaBase, // Based on Llama 3.1
	"nous-hermes2":         R50kBase,        // MIGRATION: Based on Llama 2 (SentencePiece)
	"nous-hermes2-mixtral": R50kBase,        // MIGRATION: Based on Mixtral (SentencePiece)
	"nous-hermes":          R50kBase,        // MIGRATION: Based on Llama 2 (SentencePiece)
	"openhermes":           R50kBase,        // MIGRATION: Based on Mistral (SentencePiece)

	// Wizard family
	"wizardlm2":                R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"wizardlm":                 R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"wizardlm-uncensored":      R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"wizardcoder":              R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"wizard-math":              R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"wizard-vicuna-uncensored": R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"wizard-vicuna":            R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
}

// Everything else - fallback models
var fallbackModels = map[string]Encoding{
	"starcoder2":          R50kBase, // Custom StarCoder tokenizer
	"starcoder":           R50kBase,
	"orca-mini":           R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"orca2":               R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"yi":                  R50kBase, // Custom Yi tokenizer
	"yi-coder":            R50kBase,
	"zephyr":              R50kBase, // MIGRATION: Based on Mistral (SentencePiece)
	"command-r":           R50kBase, // Cohere custom
	"command-r-plus":      R50kBase,
	"command-r7b":         R50kBase,
	"command-r7b-arabic":  R50kBase,
	"command-a":           R50kBase,
	"vicuna":              R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"openchat":            R50kBase, // MIGRATION: Based on Mistral (SentencePiece)
	"olmo2":               R50kBase,
	"dbrx":                R50kBase,
	"falcon":              R50kBase,
	"falcon2":             R50kBase,
	"falcon3":             R50kBase,
	"solar":               R50kBase,
	"solar-pro":           R50kBase,
	"stablelm2":           R50kBase,
	"stablelm-zephyr":     R50kBase,
	"stable-code":         R50kBase,
	"stable-beluga":       R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"sqlcoder":            R50kBase,
	"reflection":          OllamaLlamaBase, // Based on Llama 3.1
	"starling-lm":         R50kBase,
	"xwinlm":              R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"phind-codellama":     R50kBase, // MIGRATION: Based on Code Llama (SentencePiece)
	"internlm2":           R50kBase,
	"yarn-llama2":         R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"yarn-mistral":        R50kBase, // MIGRATION: Based on Mistral (SentencePiece)
	"nexusraven":          R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"shieldgemma":         R50kBase, // MIGRATION: Based on Gemma (SentencePiece)
	"everythinglm":        R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"codeup":              R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"duckdb-nsql":         R50kBase,
	"magicoder":           R50kBase,
	"codebooga":           R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"bespoke-minicheck":   R50kBase,
	"tulu3":               OllamaLlamaBase, // Based on Llama 3.1
	"nuextract":           R50kBase,        // Based on Phi-3
	"megadolphin":         R50kBase,        // MIGRATION: Based on Llama 2 (SentencePiece)
	"notux":               R50kBase,        // MIGRATION: Based on Mixtral (SentencePiece)
	"open-orca-platypus2": R50kBase,        // MIGRATION: Based on Llama 2 (SentencePiece)
	"notus":               R50kBase,        // MIGRATION: Based on Zephyr/Mistral (SentencePiece)
	"goliath":             R50kBase,        // MIGRATION: Based on Llama 2 (SentencePiece)
	"alfred":              R50kBase,        // MIGRATION: Based on Llama 2 (SentencePiece)
	"neural-chat":         R50kBase,        // MIGRATION: Based on Mistral (SentencePiece)
	"samantha-mistral":    R50kBase,        // MIGRATION: Based on Mistral (SentencePiece)
	"athene-v2":           R50kBase,
	"nemotron-mini":       R50kBase,
	"nemotron":            OllamaLlamaBase, // Based on Llama 3.1
	"opencoder":           R50kBase,
	"exaone3.5":           R50kBase,
	"exaone-deep":         R50kBase,
	"aya":                 R50kBase,
	"aya-expanse":         R50kBase,
	"smallthinker":        R50kBase, // Based on Qwen 2.5
	"sailor2":             R50kBase,
	"firefunction-v2":     OllamaLlamaBase, // Based on Llama 3
	"codegeex4":           R50kBase,
	"glm4":                R50kBase,
	"meditron":            R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"medllama2":           R50kBase, // MIGRATION: Based on Llama 2 (SentencePiece)
	"reader-lm":           R50kBase,
	"r1-1776":             R50kBase, // Based on DeepSeek R1
	"marco-o1":            R50kBase,
	"openthinker":         R50kBase, // Based on DeepSeek R1
	"magistral":           R50kBase,
	"cogito":              R50kBase,
}

// Combine all maps into the main lookup
func buildModelPrefixToEncoding() map[string]Encoding {
	result := make(map[string]Encoding)

	// Add all maps in order of priority (more specific to less specific)
	for k, v := range definitiveTokenizerFamilies {
		result[k] = v
	}
	for k, v := range claudeModels {
		result[k] = v
	}
	for k, v := range deepSeekModels {
		result[k] = v
	}
	for k, v := range llamaModels {
		result[k] = v
	}
	for k, v := range qwenModels {
		result[k] = v
	}
	for k, v := range mistralModels {
		result[k] = v
	}
	for k, v := range gemmaModels {
		result[k] = v
	}
	for k, v := range phiModels {
		result[k] = v
	}
	for k, v := range visionModels {
		result[k] = v
	}
	for k, v := range graniteModels {
		result[k] = v
	}
	for k, v := range smallModels {
		result[k] = v
	}
	for k, v := range embeddingModels {
		result[k] = v
	}
	for k, v := range derivedModels {
		result[k] = v
	}
	for k, v := range fallbackModels {
		result[k] = v
	}

	return result
}

var modelPrefixToEncoding = buildModelPrefixToEncoding()

// Get returns a new instance of a Codec implementation based on the specified
// encoding format. The returned Codec instance can be used to encode (tokenize)
// and decode (reassemble) text. If the specified encoding is not supported,
// an error is returned.
func Get(encoding Encoding) (Codec, error) {
	switch encoding {
	case O200kBase:
		return codec.NewO200kBase(), nil
	case Cl100kBase:
		return codec.NewCl100kBase(), nil
	case R50kBase:
		return codec.NewR50kBase(), nil
	case P50kBase:
		return codec.NewP50kBase(), nil
	case P50kEdit:
		return codec.NewP50kEdit(), nil
	case OllamaLlamaBase:
		return codec.NewLLama3Base(), nil
	// large margin of error
	// case AnthropicBase:
	// 	return codec.NewAnthropicBase(), nil
	default:
		return nil, ErrEncodingNotSupported
	}
}

// ForModel returns a new instance of a Codec implementation based on the
// specified OpenAI model. If the specified model is not supported, an error
// is returned.
func ForModel(model Model) (Codec, error) {
	switch model {
	case O1, O1Preview, O1Mini, GPT41, GPT4o, O3, O3Mini, O4Mini:
		return Get(O200kBase)

	case GPT4, GPT35, GPT35Turbo, TextEmbeddingAda002:
		return Get(Cl100kBase)

	case TextDavinci003, TextDavinci002, CodeDavinci001,
		CodeDavinci002, CodeCushman002, CodeCushman001,
		DavinciCodex, CushmanCodex:
		return Get(P50kBase)

	case TextDavinci001, TextCurie001, TextBabbage001, TextAda001, Davinci,
		Curie, Babbage, Ada, TextSimilarityDavinci001, TextSimilarityCurie001,
		TextSimilarityBabbage001, TextSimilarityAda001, TextSearchDavinciDoc001,
		TextSearchCurieDoc001, TextSearchAdaDoc001, TextSearchBabbageDoc001,
		CodeSearchBabbageCode001, CodeSearchAdaCode001:
		return Get(R50kBase)

	case TextDavinciEdit001, CodeDavinciEdit001:
		return Get(P50kEdit)

	default:
		for prefix, enc := range modelPrefixToEncoding {
			if strings.HasPrefix(string(model), string(prefix)) {
				// panic("found prefix: " + prefix + " for model: " + string(model))
				return Get(enc)
			}
		}
		return nil, ErrModelNotSupported
	}
}

func Count(model Model, input string) (int, error) {
	enc, err := ForModel(model)
	if err != nil {
		return 0, err
	}
	count, err := enc.Count(input)
	if err != nil {
		return 0, err
	}

	// account ratios
	for prefix, ratio := range Ratios {
		if strings.HasPrefix(string(model), string(prefix)) {
			count = int(float64(count) * ratio)
			break
		}
	}

	return count, nil
}
