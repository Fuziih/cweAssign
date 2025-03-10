#!/usr/bin/env python3

import connexion
import os
from swagger_server import encoder
from transformers import RobertaTokenizer

from swagger_server.cweAssign import CWEInferenceModule, CweFinder, get_label_data


def main():
    port = os.getenv("CWE_PORT", 8080)
    data_path = os.getenv("CWE_DATA_FOLDER", "/workspace/swagger_server/data/")

    app = connexion.App(__name__, specification_dir='./swagger/')

    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('swagger.yaml', arguments={'title': 'CWE predictor - OpenAPI 3.0'}, pythonic_params=True)
    app.app.config['models'] = []
    for model_file in ['t1_vulnberta', 't2_vulnberta', 't3_vulnberta', 't4_vulnberta', 't5_vulnberta']:
        app.app.config['models'].append(CWEInferenceModule.load_from_checkpoint(f"{data_path}/{model_file}.ckpt", strict=False))

    app.app.config['preps'] = (get_label_data("1", data_path),
                              get_label_data("2", data_path),
                              get_label_data("3", data_path),
                              get_label_data("4", data_path),
                              get_label_data("5", data_path))

    app.app.config['cwefinder'] = CweFinder(data_path)
    app.app.config['tokenizer'] = RobertaTokenizer.from_pretrained("roberta-base")
    app.app.logger.debug("-- API initalized --")
    app.run(port=port)


if __name__ == '__main__':
    main()
