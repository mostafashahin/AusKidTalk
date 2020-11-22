#!/usr/bin/env bash

# Copyright 2020  Mostafa Shahin (UNSW)

# This script run #TODO

PYTHON=python3.7

#Configuration
wrkDir='AKT/annotation'
HOST='auskidtalk@149.171.37.225'
DATA_DIR='/volume1/AusKidTalk_Recordings/'

#Create working directory
mkdir -p $wrkDir

childID='122' #TODO who will pass the child ID

#Check if the directory and primary file exist
FILE_PATH="$DATA_DIR/122\ 3_2_0/122\ Primary_21-01.wav"
#FILE_PATH="$DATA_DIR/122*/122*Primary*.wav"
if ssh -q $HOST [[ ! -f $FILE_PATH ]]; then
        echo "File $FILE_PATH does not exist"
        exit 1
fi


#Create child wrkDir
mkdir -p $wrkDir/$childID

echo "Download primary wav file of child $childID"
#scp -T $HOST:"$FILE_PATH" $wrkDir/$childID/primary.wav

#Convert to 16 bit
echo "Converting to 16 bit for beep detection"
#sox $wrkDir/$childID/primary.wav -c 1 -b 16 $wrkDir/$childID/primary_16.wav || exit 1

$PYTHON Initiate_Alignment/InitAlign.py $childID $wrkDir/$childID/primary_16.wav $wrkDir/$childID/txtgrids || exit 1
