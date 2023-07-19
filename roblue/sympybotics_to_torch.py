#!/usr/bin/env python3
#usage: <filename>.py <infile> <outfile>

import sys

#read  file specified in first argument into string:
with open(sys.argv[1], 'r') as f: text = f.read()
assert "from torch" not in text, "the file looks already converted because we found \"from torch\""
text = "from torch import *\n" + text
text = text. \
    replace("_out = [0]*36","_out = zeros((q.shape[0],36), device='cuda')"). \
    replace("_out = [0]*6","_out = zeros((q.shape[0],6), device='cuda')"). \
    replace('_out[', '_out[:,'). \
    replace("q[","q[:,")

with open(sys.argv[2], 'w') as f: f.write(text)

