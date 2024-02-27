##
## Extracts features for a MEM language model model:
## Given a trigram, features are extracted from the first two element,
## aiming to predict the third one.
##
## Usage:  python3 feature-extractor.py <corpus.txt >features.dat

import sys
from trigram_fex import *

## ---------------------------------
def ngrams(f,n) :
  ng = "._"
  ch = f.read(1).lower()
  if ch.isspace() : ch = '_'
  ng += ch
  while ch :
    yield ng
    ch = f.read(1).lower()
    if ch.isspace() : ch = '_'
    ng = (ng+ch)[1:]


## MAIN ---------------------

## -- read text in ngrams, and create a vector feature for each
for ng in ngrams(sys.stdin, 3) :
  print(ng[2], " ".join(extract_features(ng)))
  

