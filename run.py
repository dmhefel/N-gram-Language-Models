#!/bin/sh

# You may find this shell script helpful.
# Depending on which version(s) of Python you have installed, you may need to
# change "python3" to "python" below.

python tester.py --model unigram.Unigram --data data --train train-data.txt --test dev-data.txt --showguesses False --jumble True --generate True --check True
