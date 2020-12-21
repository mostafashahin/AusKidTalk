FROM ubuntu

RUN apt-get update

RUN apt-get -y install git python3.7 sox python3-venv python3-pip

RUN pip3 install -U pip

#git the code repository
RUN git clone https://github.com/mostafashahin/AusKidTalk.git /opt/AusKidTalk/ && \
    cd /opt/AusKidTalk && \
    pip3 install -r requirements.txt

#git the pyAudioAnalysis tool in tools directory
RUN git clone https://github.com/tyiannak/pyAudioAnalysis.git /opt/AusKidTalk/tools/pyAudioAnalysis && \
    cd /opt/AusKidTalk/tools/pyAudioAnalysis && \
    pip3 install -e .

CMD echo "Installation complete"

