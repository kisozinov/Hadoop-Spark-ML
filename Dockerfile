FROM bde2020/spark-worker:3.0.0-hadoop3.2

RUN apk add --no-cache --virtual .build-deps \
    wget \
    build-base \
    libffi-dev \
    openssl-dev \
    bzip2-dev \
    zlib-dev \
    xz-dev \
    readline-dev \
    sqlite-dev \
    libgfortran

RUN wget https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tgz && \
    tar xzf Python-3.9.18.tgz

RUN cd Python-3.9.18 && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j$(nproc) && \
    make altinstall

RUN ln -sf /usr/local/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.9 /usr/bin/pip3

RUN python3.9 -m venv /opt/venv
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel
RUN /opt/venv/bin/pip install numpy==1.21.1 pandas tqdm psutil matplotlib requests
# RUN /opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
# RUN /opt/venv/bin/pip install transformers

ENV PATH="/opt/venv/bin:$PATH"

COPY ./hadoop.env /hadoop.env
ENV INIT_DAEMON_STEP=setup_spark
