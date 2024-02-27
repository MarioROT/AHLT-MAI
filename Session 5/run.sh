#! /bin/bash

AHLT=../../..

# train NN
echo "Training NN"
python3 train.py $AHLT/data/train $AHLT/data/devel 10 mymodel

# run model on devel data and compute performance
echo "Predicting"
python3 predict.py mymodel $AHLT/data/devel > devel.out 

# evaluate results
echo "Evaluating results..."
python3 $AHLT/util/evaluator.py NER $AHLT/data/devel devel.out > devel.stats
