#!/bin/env bash

#DIR=/opt/AusKidTalk_Recordings/annotate1
tasks="task1 task2 task5"

DIR=$1
set -e
#TODO: Out more info

ls $DIR | while read direct
do
    stage=0

    childID=$direct

    LOCAL_OUT_DIR=$DIR/$direct
    [ -f $LOCAL_OUT_DIR/stage ] && stage=`cat $LOCAL_OUT_DIR/stage`

    [ $stage -ne 2 ] && continue
    
    echo $direct
    OUT_DIR=$LOCAL_OUT_DIR/asr/data
    mkdir -p $OUT_DIR
    mode=''
    for task in $tasks; do
        python3 tools/prep_data_from_txtgrid.py $LOCAL_OUT_DIR/txtgrids/primary_16b_$task.wav \
        $LOCAL_OUT_DIR/txtgrids/primary_16b_$task.txtgrid $OUT_DIR $mode -sid $childID -rid ${childID}_$task -p Prompt
        mode='-a'
    done

    echo 3 > $LOCAL_OUT_DIR/stage
done
