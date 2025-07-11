package codec

import (
	"github.com/dlclark/regexp2"
)

const (
	// GPT-2 pattern - simpler than Llama3, focuses on preserving whitespace differently
	gpt2PatStr = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
)

func NewGPT2Base() *Codec {
	gpt2BaseVocabOnce.Do(gpt2BaseVocabInit)

	splitRegexp := regexp2.MustCompile(gpt2PatStr, regexp2.None)

	// Most GPT-2 models use these basic special tokens
	specialTokens := map[string]uint{
		"<|endoftext|>": 50256, // Standard GPT-2 EOS token
		// Modern models may add more, but this is the core one
	}

	return &Codec{
		name:          "gpt2",
		vocabulary:    gpt2BaseVocab,
		splitRegexp:   splitRegexp,
		specialTokens: specialTokens,
	}
}
