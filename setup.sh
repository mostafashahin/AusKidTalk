#!/usr/bin/env bash

sudo apt-get update
sudo apt-get -y install git

sudo apt-get -y install python3.7
sudo apt-get -y install sox
sudo apt-get -y install python3-venv python3-pip
pip3 install -U pip


#git the code repository
git clone https://github.com/mostafashahin/AusKidTalk.git
cd AusKidTalk
pip3 install -r requirements.txt
#git the pyAudioAnalysis tool in tools directory
cd tools
git clone https://github.com/tyiannak/pyAudioAnalysis.git
cd pyAudioAnalysis
pip3 install -e .
cd ../..

echo "Installation complete"
