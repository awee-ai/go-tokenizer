# Tokenizer

Fork of github.com/awee-ai/go-tokenizer with (currently WIP) incorrect Ollama model tokenizer mappings.

Maybe something will get done here.

https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained

### Anthropic
python3 internal/anthropic/anthropic.py internal/anthropic/tokenizer.json internal/anthropic/anthropic.tiktoken

go run ./internal/cmd/vocab.go -encoding anthropic