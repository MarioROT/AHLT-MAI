#! /bin/bash

AHLT=../

# $AHLT/util/corenlp-server.sh -quiet true -port 9000 -timeout 15000  &
#  sleep 1
echo "Extracting features devel..."
python extract-features.py $AHLT/data/devel/ > devel.feat &
echo "Extracting features train..."
python extract-features.py $AHLT/data/train/ > train.feat

# # kill `cat /tmp/corenlp-server.running`

############################################
# train MEM model
echo "Training MEM model..."
python train.py train.feat model.mem C=10
# run MEM model
echo "Running MEM model..."
python predict.py devel.feat model.mem > devel-MEM.out
# evaluate MEM results
echo "Evaluating MEM results..."
python $AHLT/util/evaluator.py DDI $AHLT/data/devel devel-MEM.out > devel-MEM.stats
############################################

# train SVM model
echo "Training SVM model..."
python train.py train.feat model.svm C=10
# run SVM model
echo "Running SVM model..."
python predict.py devel.feat model.svm > devel-SVM.out
# evaluate SVM results
echo "Evaluating SVM results..."
python $AHLT/util/evaluator.py DDI $AHLT/data/devel devel-SVM.out > devel-SVM.stats

############################################
