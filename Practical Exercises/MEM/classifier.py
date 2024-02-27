
## usage:   cat corpus.dat | python3 classifier.py f0.mem threshold >corpus.out

import sys

from MEmodel import *



## MAIN ---------------------

model = MEmodel(sys.argv[1])

ntot=0
nokA=0

## classify each input example
line=sys.stdin.readline()
while (line!="") :
  line = line.strip().split()

  ## keep right answer, for evaluation
  tagOK = line.pop(0)

  dist = model.prob_dist_z(line)

  # find maximum
  best = None
  mx = 0
  for c in dist :
    if dist[c] > mx :
       mx = dist[c]
       best = c

  ## output chosen class and compute evaluation statistics
  print(best, end="")

  if (best==tagOK):
     nokA=nokA+1

  ntot=ntot+1

  ## output all probabilities
  for c in dist :
    print(" {:.10g}".format(dist[c]), end="")
  print("")

  ## next example
  line=sys.stdin.readline()

print("Accuracy={:5.2f}% ({:d}/{:d})".format(100*nokA/ntot, nokA, ntot))
