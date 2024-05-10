#! /bin/bash

AHLT=../

# $AHLT/util/corenlp-server.sh -quiet true -port 9000 -timeout 15000  &
# #  sleep 1
# echo "Extracting features devel..."
# python extract-features.py $AHLT/data/devel/ > devel.feat &
# echo "Extracting features train..."
# python extract-features.py $AHLT/data/train/ > train.feat
# echo "Extracting features test..."
# python extract-features.py $AHLT/data/test/ > test.feat

# # kill `cat /tmp/corenlp-server.running`

# ############################################
# train MEM model
echo "Training MEM model..."
python train.py train.feat model.mem C=10
# # run MEM model
echo "Running MEM model..."
python predict.py devel.feat model.mem > devel-MEM.out
# evaluate MEM results
echo "Evaluating MEM results..."
python $AHLT/util/evaluator.py DDI $AHLT/data/devel devel-MEM.out > devel-MEM.stats

# echo "Running MEM model on test..."
# python predict.py test.feat model.mem > test-MEM.out
# # evaluate MEM results
# echo "Evaluating MEM results..."
# python $AHLT/util/evaluator.py DDI $AHLT/data/test test-MEM.out > test-MEM.stats
# ############################################

# train SVM model
# echo "Training SVM model..."
# python train.py train.feat model.svm C=10
# # run SVM model
# echo "Running SVM model..."
# python predict.py devel.feat model.svm > devel-SVM.out
# # evaluate SVM results
# echo "Evaluating SVM results..."
# python $AHLT/util/evaluator.py DDI $AHLT/data/devel devel-SVM.out > devel-SVM.stats

# echo "Running SVM model on test..."
# python predict.py test.feat model.svm > test-SVM.out
# # evaluate SVM results
# echo "Evaluating SVM results..."
# python $AHLT/util/evaluator.py DDI $AHLT/data/test test-SVM.out > test-SVM.stats

############################################

## VERY BAD
# # train AB model
# echo "Training AB model..."
# python train.py train.feat model.abc
# # run AB model
# echo "Running AB model..."
# python predict.py devel.feat model.abc > devel-AB.out
# # evaluate AB results
# echo "Evaluating AB results..."
# python $AHLT/util/evaluator.py DDI $AHLT/data/devel devel-AB.out > devel-AB.stats

############################################
