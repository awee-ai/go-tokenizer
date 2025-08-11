#!/usr/bin/env python3

import json
import base64
import sys

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

vocab = data['model']['vocab']

startIndex = 0

with open(sys.argv[2], 'w') as f:
    for token, rank in sorted(vocab.items(), key=lambda x: x[1]):
        if rank >= startIndex:
            token_b64 = base64.b64encode(token.encode('utf-8')).decode('ascii')
            f.write(f"{token_b64} {rank - startIndex}\n")