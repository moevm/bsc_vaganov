FROM tknerr/baseimage-ubuntu:18.04
ADD ./* /
WORKDIR /bsc_vaganov
COPY test_stand/ .
COPY test_stand/requirements.txt .
COPY test_stand/start.py .
COPY test_stand/app.py .
COPY data_collect_and_preprocess/ .
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && pip3 install --upgrade setuptools
RUN apt-get -y install curl && pip3 install -r requirements.txt
COPY . .
EXPOSE 5001
ENTRYPOINT [ "python3" ]
CMD [ "start.py" ]