package codec

import (
	"fmt"

	"github.com/dlclark/regexp2"
)

const (
	numReservedSpecialTokens = 256
	// Llama3 pattern string - this is the regex pattern used for tokenization
	llamaPatStr = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
)

func NewLLama3Base() *Codec {
	llamaVocabOnce.Do(llamaVocabInit)

	splitRegexp := regexp2.MustCompile(llamaPatStr, regexp2.None)

	numBaseTokens := len(llamaVocab)

	specialTokens := map[string]uint{
		"<|begin_of_text|>":            uint(numBaseTokens),
		"<|end_of_text|>":              uint(numBaseTokens + 1),
		"<|reserved_special_token_0|>": uint(numBaseTokens + 2),
		"<|reserved_special_token_1|>": uint(numBaseTokens + 3),
		"<|finetune_right_pad_id|>":    uint(numBaseTokens + 4),
		"<|step_id|>":                  uint(numBaseTokens + 5),
		"<|start_header_id|>":          uint(numBaseTokens + 6),
		"<|end_header_id|>":            uint(numBaseTokens + 7),
		"<|eom_id|>":                   uint(numBaseTokens + 8), // end of message
		"<|eot_id|>":                   uint(numBaseTokens + 9), // end of turn
		"<|python_tag|>":               uint(numBaseTokens + 10),
		"<|image|>":                    uint(numBaseTokens + 11),
	}

	definedSpecialTokens := 12
	for i := 0; i < (numReservedSpecialTokens - definedSpecialTokens); i++ {
		tokenName := fmt.Sprintf("<|reserved_special_token_%d|>", 2+i)
		specialTokens[tokenName] = uint(numBaseTokens + definedSpecialTokens + i)
	}

	return &Codec{
		name:          "llama",
		vocabulary:    llamaVocab,
		splitRegexp:   splitRegexp,
		specialTokens: specialTokens,
	}
}
