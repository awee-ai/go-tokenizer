package tokenizer_test

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/awee-ai/go-tokenizer"
)

type testCase struct {
	text string
	ids  []uint
}
type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type anthropic struct {
	Model    string    `json:"model"`
	Messages []message `json:"messages"`
}
type testCase2 struct {
	name                string
	model               string
	text                string
	body                anthropic
	expectedRemoteCount int
	expectedLocalCount  int
}
type anthropicResponse struct {
	InputTokens int `json:"input_tokens"`
}

// model=claude-opus-4-1-20250805
// model=claude-opus-4-20250514
// model=claude-sonnet-4-20250514
// model=claude-3-7-sonnet-20250219
// model=claude-3-5-sonnet-20241022
// model=claude-3-5-haiku-20241022
// model=claude-3-5-sonnet-20240620
// model=claude-3-haiku-20240307
// model=claude-3-opus-20240229
func Test_AnthropicModel_TokenCounts(t *testing.T) {
	url := "https://api.anthropic.com/v1/messages/count_tokens"

	body := anthropic{
		Messages: []message{
			{
				Role:    "user",
				Content: "This tool uses Anthropic's newly released token counting api to count the number of tokens in a given text. Beware of existing tokenizers which are not accurate. Explore the source code here.",
			},
		},
	}
	body2 := anthropic{
		Messages: []message{
			{
				Role: "user",
				Content: "This tool uses Anthropic's newly released token counting api to count the number of tokens in a given text. Beware of existing tokenizers which are not accurate. Explore the source code here." +
					" This tool uses Anthropic's newly released token counting api to count the number of tokens in a given text. Beware of existing tokenizers which are not accurate. Explore the source code here.",
			},
		},
	}

	tests := []testCase2{
		{
			name:                "claude-3-7-sonnet-20250219",
			model:               "claude-3-7-sonnet-20250219",
			body:                body2,
			expectedRemoteCount: 91,
			expectedLocalCount:  93,
		},
		{
			name:                "claude-opus-4-1-20250805",
			model:               "claude-opus-4-1-20250805",
			body:                body2,
			expectedRemoteCount: 91,
			expectedLocalCount:  93,
		},
		// diff 28
		{
			name:                "claude-opus-4-20250514",
			model:               "claude-opus-4-20250514",
			body:                body2,
			expectedRemoteCount: 91,
			expectedLocalCount:  93,
		},
		{
			name:                "claude-sonnet-4-20250514",
			model:               "claude-sonnet-4-20250514",
			body:                body,
			expectedRemoteCount: 49,
			expectedLocalCount:  47,
		},
		{
			name:                "claude-3-7-sonnet-20250219",
			model:               "claude-3-7-sonnet-20250219",
			body:                body,
			expectedRemoteCount: 49,
			expectedLocalCount:  47,
		},
		{
			name:                "claude-3-5-sonnet-20241022",
			model:               "claude-3-5-sonnet-20241022",
			body:                body,
			expectedRemoteCount: 49,
			expectedLocalCount:  47,
		},
		{
			name:                "claude-3-5-haiku-20241022",
			model:               "claude-3-5-haiku-20241022",
			body:                body,
			expectedRemoteCount: 49,
			expectedLocalCount:  47,
		},
	}

	keys, err := readEnvFile(".env")
	assert.NoError(t, err, "failed to read environment variables from .env file")
	key, ok := keys["ANTHROPIC_API_KEY"]
	assert.True(t, ok, "ANTHROPIC_API_KEY not found in environment variables")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.body.Model = tt.model
			c, err := json.Marshal(tt.body)
			if err != nil {
				fmt.Println("error marshalling JSON:", err)
			}
			payload := strings.NewReader(string(c))

			req, err := http.NewRequest("POST", url, payload)
			assert.NoError(t, err, "failed to create HTTP request")

			req.Header.Add("anthropic-version", "2023-06-01")
			req.Header.Add("x-api-key", key)
			req.Header.Add("Content-Type", "application/json")

			res, err := http.DefaultClient.Do(req)
			assert.NoError(t, err, "failed to send HTTP request")

			defer res.Body.Close()
			body, err := io.ReadAll(res.Body)
			assert.NoError(t, err, "failed to read response body")
			fmt.Println("response body:", string(body))
			response := anthropicResponse{}
			err = json.Unmarshal(body, &response)
			assert.NoError(t, err, "failed to unmarshal response body")
			assert.Equal(t, tt.expectedRemoteCount, response.InputTokens, "Remote token count mismatch for model %s", tt.model)

			count, err := tokenizer.Count(tokenizer.Model(tt.model), tt.body.Messages[0].Role+": "+tt.body.Messages[0].Content)
			assert.NoError(t, err, "failed to locally count tokens for model %s", tt.model)
			assert.Equal(t, tt.expectedLocalCount, count, "Local token count mismatch for model %s", tt.model)
		})
	}
}

func Test_Model_TokenCount(t *testing.T) {
	tests := []testCase2{
		// {
		// 	name:          "gpt-5",
		// 	model:         "gpt-5",
		// 	text:          "OpenAI's large language models process text using tokens, which are common sequences of characters found in a set of text. The models learn to understand the statistical relationships between these tokens, and excel at producing the next token in a sequence of tokens. Learn more.",
		// 	expectedRemoteCount: 52,
		// },
		{
			name:                "claude-opus-4",
			model:               "claude-opus-4",
			text:                "This tool uses Anthropic's newly released token counting api to count the number of tokens in a given text. Beware of existing tokenizers which are not accurate. Explore the source code here.",
			expectedRemoteCount: 45,
		},
		{
			name:                "gemini-1.5-pro",
			model:               "gemini-1.5-pro",
			text:                "This tool uses Anthropic's newly released token counting api to count the number of tokens in a given text. Beware of existing tokenizers which are not accurate. Explore the source code here.",
			expectedRemoteCount: 39,
		},
		{
			name:                "gemma-3-4b",
			model:               "gemma-3-4b",
			text:                "This tool uses Anthropic's newly released token counting api to count the number of tokens in a given text. Beware of existing tokenizers which are not accurate. Explore the source code here.",
			expectedRemoteCount: 38,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			count, err := tokenizer.Count(tokenizer.Model(tt.model), tt.text)
			assert.NoError(t, err, "failed to count tokens for model %s", tt.model)
			assert.Equal(t, tt.expectedRemoteCount, count, "Token count mismatch for model %s", tt.model)
		})
	}
}

func TestO200kBase(t *testing.T) {
	tok, err := tokenizer.Get(tokenizer.O200kBase)
	if err != nil {
		t.Fatalf("can't create tokenizer: %v", err)
	}

	tests := []testCase{
		{text: "hello world", ids: []uint{24912, 2375}},
		{text: "hello  world", ids: []uint{24912, 220, 2375}},
		{text: "hello   world", ids: []uint{24912, 256, 2375}},
		{text: "supercalifragilistic", ids: []uint{17789, 5842, 366, 17764, 311, 6207}},
		{text: "We know what we are, but know not what we may be.", ids: []uint{2167, 1761, 1412, 581, 553, 11, 889, 1761, 625, 1412, 581, 1340, 413, 13}},
	}

	runTests(t, tok, tests)
}

func TestCl100kBase(t *testing.T) {
	tok, err := tokenizer.Get(tokenizer.Cl100kBase)
	if err != nil {
		t.Fatalf("can't create tokenizer: %v", err)
	}

	tests := []testCase{
		{text: "hello world", ids: []uint{15339, 1917}},
		{text: "hello  world", ids: []uint{15339, 220, 1917}},
		{text: "hello   world", ids: []uint{15339, 256, 1917}},
		{text: "supercalifragilistic", ids: []uint{13066, 3035, 278, 333, 4193, 321, 4633}},
		{text: "We know what we are, but know not what we may be.", ids: []uint{1687, 1440, 1148, 584, 527, 11, 719, 1440, 539, 1148, 584, 1253, 387, 13}},
	}

	runTests(t, tok, tests)
}

func TestR50kBase(t *testing.T) {
	tok, err := tokenizer.Get(tokenizer.R50kBase)
	if err != nil {
		t.Fatalf("can't create tokenizer: %v", err)
	}

	tests := []testCase{
		{text: "hello world", ids: []uint{31373, 995}},
		{text: "hello  world", ids: []uint{31373, 220, 995}},
		{text: "hello   world", ids: []uint{31373, 220, 220, 995}},
		{text: "supercalifragilistic", ids: []uint{16668, 9948, 361, 22562, 346, 2569}},
		{text: "We know what we are, but know not what we may be.", ids: []uint{1135, 760, 644, 356, 389, 11, 475, 760, 407, 644, 356, 743, 307, 13}},
	}

	runTests(t, tok, tests)
}

func TestP50kBase(t *testing.T) {
	tok, err := tokenizer.Get(tokenizer.P50kBase)
	if err != nil {
		t.Fatalf("can't create tokenizer: %v", err)
	}

	tests := []testCase{
		{text: "hello world", ids: []uint{31373, 995}},
		{text: "hello  world", ids: []uint{31373, 220, 995}},
		{text: "hello   world", ids: []uint{31373, 50257, 995}},
		{text: "supercalifragilistic", ids: []uint{16668, 9948, 361, 22562, 346, 2569}},
		{text: "We know what we are, but know not what we may be.", ids: []uint{1135, 760, 644, 356, 389, 11, 475, 760, 407, 644, 356, 743, 307, 13}},
	}

	runTests(t, tok, tests)
}

func runTests(t *testing.T, tok tokenizer.Codec, tests []testCase) {
	for _, test := range tests {
		t.Run(test.text, func(t *testing.T) {
			ids, _, err := tok.Encode(test.text)
			if err != nil {
				t.Fatalf("error encoding: %v", err)
			}
			if !sliceEqual(ids, test.ids) {
				t.Errorf("encoding mismatch - want: %v got: %v", test.ids, ids)
			}

			text, err := tok.Decode(ids)
			if err != nil {
				t.Fatalf("error decoding: %v", err)
			}
			if text != test.text {
				t.Errorf("decoding mismatch - want: %s got: %s", test.text, text)
			}

			count, err := tok.Count(test.text)
			if err != nil {
				t.Fatalf("error counting: %v", err)
			}
			if count != len(test.ids) {
				t.Errorf("count mismatch - want: %d got: %d", len(test.ids), count)
			}
		})
	}
}

func sliceEqual(a, b []uint) bool {
	if len(a) != len(b) {
		return false
	}
	for i, elem := range a {
		if elem != b[i] {
			return false
		}
	}
	return true
}

func readEnvFile(path string) (map[string]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open env file: %w", err)
	}
	defer file.Close()

	env := make(map[string]string)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue // skip empty lines and comments
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid line in env file: %s", line)
		}
		env[parts[0]] = parts[1]
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading env file: %w", err)
	}

	return env, nil
}
