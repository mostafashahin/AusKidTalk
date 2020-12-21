#!/usr/bin/env bash

# Copyright 2020  Mostafa Shahin (UNSW)

# This script run #TODO

PYTHON=/usr/bin/python3


DIR=/opt/AusKidTalk_Recordings/
OUT_DIR=$DIR/annotate1
stage=0

#TODO list directories only
ls $DIR | while read direct
do

    stage=0

    echo "Checking Directory $direct"

    WAV_FILE=`ls  "$DIR/$direct/" | grep Primary`

    [ -z "$WAV_FILE" ] && echo "ERROR: Primary wav file not exist!" && continue

    childID=`echo "$WAV_FILE" | cut -d' ' -f1`

    echo "Child ID $childID"

    #TODO check if childID already processed //DONE

    WAV_FILE_PATH=$DIR/$direct/$WAV_FILE

    LOCAL_OUT_DIR=$OUT_DIR/$childID

    if [ -f $LOCAL_OUT_DIR/stage ]; then
        stage=`cat $LOCAL_OUT_DIR/stage`
        [ -z $stage ] && stage=0
    fi

    echo "Starting child $childID from stage $stage"

    if [ $stage -le 0 ]; then

        mkdir -p "$LOCAL_OUT_DIR"

        #Convert to 16 bit
        echo "Converting to 16 bit for beep detection"
        sox "$WAV_FILE_PATH" -c 1 -b 16 "$LOCAL_OUT_DIR/primary_16b.wav" || continue
        #TODO validate converted wav file, same duration of the original one
        #TODO if converted file exist don't convert again
        echo 1 > $LOCAL_OUT_DIR/stage
    fi

    if [ $stage -le 1 ]; then
        cd /opt/AusKidTalk/
        ! $PYTHON Initiate_Alignment/InitAlign.py $childID "$LOCAL_OUT_DIR/primary_16b.wav" $LOCAL_OUT_DIR/txtgrids && mv InitAlign.log "$LOCAL_OUT_DIR" && echo 100 > $LOCAL_OUT_DIR/stage && continue
        mv InitAlign.log "$LOCAL_OUT_DIR"
        echo 2 > $LOCAL_OUT_DIR/stage
    fi
    echo $childID

done

