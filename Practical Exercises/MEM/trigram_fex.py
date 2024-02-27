

## -------------------------------
## extract ME features from a trigram

def extract_features(xy) :
  return ["1st:"+xy[0], "2nd:"+xy[1], "bigr:"+xy[0:2] ]     


