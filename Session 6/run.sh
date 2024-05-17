#! /bin/bash

AHLT=../

if [[ "$*" == *"parse"* ]]; then
   # $AHLT/util/corenlp-server.sh -quiet true -port 9000 -timeout 15000 &
   # sleep 1

   PYTHONPATH=$AHLT/util
   python parse_data.py $AHLT/data/train train
   python parse_data.py $AHLT/data/devel devel
   # kill `cat /tmp/corenlp-server.running`
fi

if [[ "$*" == *"train"* ]]; then
    rm -rf model model.idx
    python train.py train.pck devel.pck 1 model
fi

if [[ "$*" == *"predict"* ]]; then
   rm -f devel.stats devel.out
   python predict.py model devel.pck > devel.out 
   python $AHLT/util/evaluator.py DDI $AHLT/data/devel devel.out > devel.stats
fi


