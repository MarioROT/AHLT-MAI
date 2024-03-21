#! /bin/bashLR

BASEDIR=./

# # convert datasets to feature vectors
# echo "Extracting features..."
# python extract-features.py ../data/train/ > train.feat
# python extract-features.py ../data/devel/ > devel.feat

# # train CRF model
# echo "Training CRF model..."
# python train.py train.feat model.crf 
# # run CRF model
# echo "Running CRF model..."
# python predict.py devel.feat model.crf > devel-CRF.out
# # evaluate CRF results
# echo "Evaluating CRF results..."
# python ../util/evaluator.py NER ../data/devel devel-CRF.out > devel-CRF.stats
# python ../util/evaluator.py NER ../data/ devel-CRF.out NER-ML-CRF > devel-CRF.stats

# # train LR model
# echo "Training LR model..."
# python train.py train.feat model.lrg 
# # run LR model
# echo "Running LR model..."
# python predict.py devel.feat model.lrg > devel-LR.out
# # evaluate LR results
# echo "Evaluating LR results..."
# python ../util/evaluator.py NER ../data/devel devel-LR.out > devel-LR.stats
# python ../util/evaluator.py NER ../data/ devel-LR.out NER-ML-LR > devel-LR.stats


# # train SVM model
# echo "Training SVM model..."
# python train.py train.feat model.svm 
# # run SVM model
# echo "Running SVM model..."
# python predict.py devel.feat model.svm > devel-SVM.out
# # evaluate SVM results
# echo "Evaluating SVM results..."
# python ../util/evaluator.py NER ../data/devel devel-SVM.out > devel-SVM.stats
# python ../util/evaluator.py NER ../data/ devel-SVM.out NER-ML-SVM > devel-SVM.stats


# train RF model
echo "Training RF model..."
python train.py train.feat model.rft 
# run RF model
echo "Running RF model..."
python predict.py devel.feat model.rft > devel-RF.out
# evaluate RF results
echo "Evaluating RF results..."
python ../util/evaluator.py NER ../data/devel devel-RF.out > devel-RF.stats
# python ../util/evaluator.py NER ../data/ devel-RF.out NER-ML-RF > devel-RF.stats

# # train AdaBoost model
# echo "Training AdaBoost model..."
# python train.py train.feat model.abc 
# # run AdaBoost model
# echo "Running AdaBoost model..."
# python predict.py devel.feat model.abc > devel-AB.out
# # evaluate AdaBoost results
# echo "Evaluating AdaBoost results..."
# python ../util/evaluator.py NER ../data/devel devel-AB.out > devel-AB.stats

