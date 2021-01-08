FROM auskidtalk/asr_base

RUN apt-get update && \
    apt-get install -y git sox python3-venv python3-pip curl

RUN pip3 install -U pip

ENV DOCKERVERSION=20.10.1

RUN curl -fsSLO https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKERVERSION}.tgz \
    && tar xzvf docker-${DOCKERVERSION}.tgz --strip 1 -C /usr/local/bin docker/docker \
    && rm docker-${DOCKERVERSION}.tgz

RUN git clone https://github.com/mostafashahin/AusKidTalk.git /opt/AusKidTalk/ && \
    cd /opt/AusKidTalk && \
    pip3 install -r requirements.txt

RUN cd /opt/kaldi/egs && \
    git clone https://github.com/mostafashahin/akt-asr.git
