$AHLT = "../"

if ($args -contains "parse") {
    # Start-Process -FilePath "$AHLT/util/corenlp-server.sh" -ArgumentList "-quiet true -port 9000 -timeout 15000"
    # Start-Sleep -Seconds 1

    $env:PYTHONPATH = "$AHLT/util"
    python parse_data.py "$AHLT/data/train" "train"
    python parse_data.py "$AHLT/data/devel" "devel"
    python parse_data.py "$AHLT/data/test" "test"
    # Stop-Process -Id (Get-Content /tmp/corenlp-server.running)
}

if ($args -contains "train") {
    # Remove-Item -Recurse -Force model, model.idx
    python train.py "train.pck" "devel.pck" 10 "model"
}

if ($args -contains "predict") {
    # Remove-Item -Force devel.stats, devel.out
    python predict.py "model" "devel.pck" > "devel-NN.out"
    python predict.py "model" "train.pck" > "train-NN.out"
    python predict.py "model" "test.pck" > "test-NN.out"
    #python "$AHLT/util/evaluator.py" DDI "$AHLT/data/devel" "devel.out" > "devel.stats"
    python $AHLT/util/evaluator.py DDI $AHLT/data/ NN.out DDI-NN > NN-tuned.stats
}