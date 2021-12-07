# Base Image
FROM tensorflow/tensorflow:2.6.0-gpu

RUN apt-get update && apt-get -q -y install git-core wget

# Installing the Tensorflow Object Detection API
RUN git clone --depth 1 https://github.com/tensorflow/models.git /tensorflow/models
# Get protoc 3.0.0, rather than the old version already in the container
RUN apt-get install -y curl unzip

RUN curl -OL "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip" && \
    unzip protoc-3.0.0-linux-x86_64.zip -d proto3 && \
    mv proto3/bin/* /usr/local/bin && \
    mv proto3/include/* /usr/local/include && \
    rm -rf proto3 protoc-3.0.0-linux-x86_64.zip

# Compile protobuf configs
RUN (cd /tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)

# Install object detection api dependencies
RUN pip install Cython && \
    pip install pycocotools

# Install pycocoapi
RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    make -j8 && \
    cp -r pycocotools /tensorflow/models/research && \
    cd ../../ && \
    rm -rf cocoapi

WORKDIR /tensorflow/models/research/

RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/tensorflow/.local/bin:${PATH}"
RUN python3 -m pip install .

# install repo packages containing helper functions
COPY src/packages/azure_utils src/packages/azure_utils
COPY src/packages/tfod_utils src/packages/tfod_utils
COPY docker/tf_2/requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
