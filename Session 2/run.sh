#! /bin/bash

BASEDIR=../../..

# convert datasets to feature vectors
echo "Extracting features..."
python3 extract-features.py $BASEDIR/data/train/ > train.feat
python3 extract-features.py $BASEDIR/data/devel/ > devel.feat

# train CRF model
echo "Training CRF model..."
python3 train.py train.feat model.crf 
# run CRF model
echo "Running CRF model..."
python3 predict.py devel.feat model.crf > devel-CRF.out
# evaluate CRF results
echo "Evaluating CRF results..."
python3 $BASEDIR/util/evaluator.py NER $BASEDIR/data/devel devel-CRF.out > devel-CRF.stats

# train LR model
echo "Training LR model..."
python3 train.py train.feat model.lrg 
# run LR model
echo "Running CRF model..."
python3 predict.py devel.feat model.lrg > devel-LR.out
# evaluate LR results
echo "Evaluating LR results..."
python3 $BASEDIR/util/evaluator.py NER $BASEDIR/data/devel devel-LR.out > devel-LR.stats

