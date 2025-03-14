# cweAssign

Repository for VulnBERTa: On automating CWE weakness assignment and improving the quality of cybersecurity CVE vulnerabilities through ML/NLP published in DevSecOps Research and Opportunities (DevSecOpsRO 2024) on July 12th, 2024 in Vienna, Austria co-located with the 9th IEEE European Symposium on Security and Privacy (EuroS&P 2024).

The API is a **proof-of-concept** with many missing features, and it does not represent any production-ready code.
## Overview
This server was generated by the [swagger-codegen](https://github.com/swagger-api/swagger-codegen) project. By using the
[OpenAPI-Spec](https://github.com/swagger-api/swagger-core/wiki) from a remote server, you can easily generate a server stub.  This
is an example of building a swagger-enabled Flask server.

This example uses the [Connexion](https://github.com/zalando/connexion) library on top of Flask.

## Requirements
Python 3.8

## Required configuration
### Setting the environment
Please set these environment variables before building the Docker image or running the server natively. If you don't, defaults will be used.

```bash
export CWE_PORT=<your port>
export CWE_DATA_FOLDER=<folder, where the models (zip or ckpt) are stored from IDA>
```

Defaults:
    CWE_PORT = 8080
    CWE_DATA_FOLDER = /workspace/swagger_server/data/


### Download data files

Download the tier models and label keys from the [Fairdata](https://etsin.fairdata.fi/dataset/738eb380-3c17-436d-b0eb-1aea0093462c). Place them into to folder that you map into the docker with $CWE_DATA_FOLDER.


## Usage

Set up the environment as explained in the configuration.

To run the server, please execute the following from the root directory:

```bash
pip3 install -r requirements.txt
python3 -m swagger_server
```

and open your browser to here:

```
http://localhost:8080/ui/
```

- change the port to what you have set up.

Your Swagger definition lives here:

```
http://localhost:8080/swagger.json
```
- change the port to what you have set up.


## Running with Docker
Set up the environment as explained in the configuration.

To run the server on a Docker container, please execute the following from the root directory:
```bash
# building the image
docker build -t cwe_assign .

# starting up a container
docker run --rm -v <your local folder with models>:$CWE_DATA_FOLDER -p $CWE_PORT:$CWE_PORT cwe_assign
```

## Results

    Example:

    "results": [
        {
        "cwe": "CWE-22",
        "confidence": 9.438360214233398,
        "predictions": "('CWE-664', 4.762214183807373), ('CWE-706', 7.594253063201904), ('CWE-22', 9.438360214233398), ('CWE-209', 2.57566237449646), ('CWE-416', 0.8431477546691895)",
        "tree": [
            "CWE-706",
            "CWE-22"
        ]
        }
    ]


    - cwe
        - highest confidence CWE out of the 5 tiers of models
    - confidence
        - confidence of the best inference
    - predictions
        - predictions from all 5 tiers and their confidences
    - tree:
        - higher tier CWEs from the hierarchy from the best result towards higher abstraction



## Attribution

Turtiainen, Hannu, and Andrei Costin. "VulnBERTa: On automating CWE weakness assignment and improving the quality of cybersecurity CVE vulnerabilities through ML/NLP." 2024 IEEE European Symposium on Security and Privacy Workshops (EuroS&PW). IEEE, 2024.

or

@inproceedings{turtiainen2024vulnberta,
  title={VulnBERTa: On automating CWE weakness assignment and improving the quality of cybersecurity CVE vulnerabilities through ML/NLP},
  author={Turtiainen, Hannu and Costin, Andrei},
  booktitle={2024 IEEE European Symposium on Security and Privacy Workshops (EuroS\&PW)},
  pages={618--625},
  year={2024},
  organization={IEEE}
}

