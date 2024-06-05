#! /bin/bash

AHLT=../

## train NN
echo "Training NN"
python train.py $AHLT/data/train $AHLT/data/devel 10 mymodel

## run model on devel data and compute performance
echo "Predicting train"
python predict.py mymodel $AHLT/data/train > train-NN.out 
echo "Predicting devel"
python predict.py mymodel $AHLT/data/devel > devel-NN.out 
echo "Predicting test"
python predict.py mymodel $AHLT/data/test > test-NN.out 

# evaluate results
echo "Evaluating results..."
# python $AHLT/util/evaluator.py NER $AHLT/data/devel devel.out > devel.stats
python $AHLT/util/evaluator.py NER $AHLT/data/ NN.out NER-NN > NN-cnn.stats
