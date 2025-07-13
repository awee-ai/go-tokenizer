#!/usr/bin/env python3

import json
import base64
import sys

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

vocab = data['model']['vocab']

with open(sys.argv[2], 'w') as f:
    for token, rank in sorted(vocab.items(), key=lambda x: x[1]):
        if rank >= 5:  # Skip first 5 special tokens
            token_b64 = base64.b64encode(token.encode('utf-8')).decode('ascii')
            f.write(f"{token_b64} {rank - 5}\n")