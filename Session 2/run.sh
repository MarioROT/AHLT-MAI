#! /bin/bash

BASEDIR=./

# convert datasets to feature vectors
echo "Extracting features..."
python extract-features.py ../data/train/ > train.feat
python extract-features.py ../data/devel/ > devel.feat

# train CRF model
echo "Training CRF model..."
python train.py train.feat model.crf 
# run CRF model
echo "Running CRF model..."
python predict.py devel.feat model.crf > devel-CRF.out
# evaluate CRF results
echo "Evaluating CRF results..."
python ../util/evaluator.py NER ../data/devel devel-CRF.out > devel-CRF.stats

# train LR model
echo "Training LR model..."
python train.py train.feat model.lrg 
# run LR model
echo "Running LR model..."
python predict.py devel.feat model.lrg > devel-LR.out
# evaluate LR results
echo "Evaluating LR results..."
python ../util/evaluator.py NER ../data/devel devel-LR.out > devel-LR.stats

