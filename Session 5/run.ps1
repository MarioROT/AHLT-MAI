# Define the path to the data
$AHLT = "../"

# Train NN
Write-Output "Training NN"
python train.py "$AHLT/data/train" "$AHLT/data/devel" 10 mymodel

# Run model on devel data and compute performance
Write-Output "Predicting train"
python predict.py mymodel "$AHLT/data/train" > train-NN.out 
Write-Output "Predicting devel"
python predict.py mymodel "$AHLT/data/devel" > devel-NN.out 
Write-Output "Predicting test"
python predict.py mymodel "$AHLT/data/test" > test-NN.out 

# Evaluate results
Write-Output "Evaluating results..."
# python "$AHLT/util/evaluator.py" NER "$AHLT/data/devel" devel.out > devel.stats
python "$AHLT/util/evaluator.py" NER "$AHLT/data/" NN.out NER-NN > NN-tuned.stats
# python "$AHLT/util/evaluator.py" NER "$AHLT/data/devel" devel-NN.out > NNTests.stats
