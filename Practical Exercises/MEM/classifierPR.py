
## usage:   cat corpus.dat | python3 classifier.py f0.mem threshold >corpus.out

import sys

from MEmodel import *


## MAIN ---------------------

model = MEmodel(sys.argv[1])

threshold=float(sys.argv[2])

ntot=0
nok=0
nokA=0
nans=0
## classify each input example
line=sys.stdin.readline()
while (line!="") :
  line = line.strip().split()

  ## keep right answer, for evaluation
  tagOK = line.pop(0)

  dist = model.prob_dist_z(line)

  # check classes above threshold. Remember maximum
  best = None
  mx = 0
  for c in dist :
    if dist[c] > mx :
       mx = dist[c]
       best = c

    #if dist[c]>threshold :
      # class c is over given threshold:
      # TODO ... output class name
      # TODO ... count how many were predicted (nans)
      # TODO ... count how many were right (nok)

  if (best == tagOK):
    nokA  =nokA+1

  ntot=ntot+1
  print("/",end="")

  ## output all probabilities
  for c in dist :
    print(" {:.10g}".format(dist[c]), end="")
  print("")

  ## next example
  line=sys.stdin.readline()

R=100*nok/ntot
P=100*nok/nans
F=2*P*R/(P+R)
print("Accuracy={:5.2f}% ({:d}/{:d})".format(100*nokA/ntot, nokA, ntot))
print("Recall={:5.2f}% ({:d}/{:d})".format(R,nok,ntot))
print("Precision={:5.2f}% ({:d}/{:d})".format(P,nok,nans))
print("F1-score={:5.2f}%".format(F))
