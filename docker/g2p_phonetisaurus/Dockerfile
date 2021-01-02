FROM ubuntu

RUN apt-get update

RUN apt-get -y install git

RUN git clone https://github.com/mostafashahin/aus_lexicon.git /opt/aus_lexicon/

RUN cd /opt/aus_lexicon/g2p/phonetisaurus/ && \ 
    apt-get install -y g++ autoconf-archive make libtool wget && \
    apt-get install -y python-setuptools python-dev && \
    apt-get install -y gfortran && \
    apt-get install -y python3-pip && \
    pip3 install pybindgen

RUN cd /opt/aus_lexicon/g2p/phonetisaurus/ && \
    wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.2.tar.gz && \
    tar -xvzf openfst-1.6.2.tar.gz
    
RUN cd /opt/aus_lexicon/g2p/phonetisaurus/openfst-1.6.2 && \
    ./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts && \
    make -j $(nproc) && \
    make install

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib:/usr/local/lib/fst"

RUN cd /opt/aus_lexicon/g2p/phonetisaurus/ && \
    git clone https://github.com/AdolfVonKleist/Phonetisaurus.git && \
    cd Phonetisaurus/ && \
    pip3 install pybindgen && \
    PYTHON=python3 ./configure --enable-python && \
    make && \
    make install && \
    cd python && \
    cp ../.libs/Phonetisaurus.so . && \
    python3 setup.py install

RUN cd /opt/aus_lexicon/g2p/phonetisaurus/ && \
    git clone https://github.com/mitlm/mitlm.git && \
    cd mitlm/ && \
    ./autogen.sh && \
    make && \
    make install

RUN cp /opt/aus_lexicon/g2p/phonetisaurus/support_p2p/* /usr/local/bin/

CMD echo "Installation complete"
