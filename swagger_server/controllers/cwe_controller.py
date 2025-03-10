import connexion
import torch

from swagger_server.models.result import Result  # noqa: E501
from swagger_server.models.sample import Sample  # noqa: E501
from flask import current_app

def get_cwe_tree(body):  # noqa: E501
    """Get CWE by sample

    Get a CWE tree prediction for the given sample. # noqa: E501

    :param body: Get CWE by sample
    :type body: dict | bytes

    :rtype: Result
    """
    if connexion.request.is_json:
        body = Sample.from_dict(connexion.request.get_json())  # noqa: E501
        preds = []
        inputs = current_app.config['tokenizer'](body.bug_text, return_tensors="pt", max_length=512, truncation=True)
        for tier, (model, prep) in enumerate(zip(current_app.config['models'], current_app.config['preps'])):
            with torch.no_grad():
                output, _ = model(**inputs)
                confidence, _index = torch.max(output, 1)
                y_pred = prep.int_labels[_index.item()]
                preds.append((y_pred, confidence.item()))

        tree = []
        best = max(preds, key=lambda x: x[1])
        tier = preds.index(best) + 1
        cwe = best[0]
        for _ in range(tier):
            if cwe:
                root = current_app.config['cwefinder'].find_root_cwe(cwe[4:])
                if root:
                    tree.append('CWE-' + root)
                cwe = root
        tree.reverse()
        tree.append(best[0])
        return Result(
            cwe=best[0],
            predictions=str(preds).strip('[]'),
            tree=tree,
            confidence=best[1]
        ), 200
    else:
        return 'Body not found.', 404
