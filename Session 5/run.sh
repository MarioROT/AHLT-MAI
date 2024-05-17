#! /bin/bash

AHLT=../

# train NN
echo "Training NN"
python train.py $AHLT/data/train $AHLT/data/devel 1 mymodel

# run model on devel data and compute performance
echo "Predicting"
python predict.py mymodel $AHLT/data/devel > devel.out 

# evaluate results
echo "Evaluating results..."
python $AHLT/util/evaluator.py NER $AHLT/data/devel devel.out > devel.stats
