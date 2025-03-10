FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV CWE_DATA_FOLDER=/workspace/swagger_server/data

WORKDIR /workspace

COPY requirements.txt /workspace

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . /workspace

RUN mkdir -p ${CWE_DATA_FOLDER}

EXPOSE 8080

ENTRYPOINT ["python3"]

CMD ["-m", "swagger_server"]
