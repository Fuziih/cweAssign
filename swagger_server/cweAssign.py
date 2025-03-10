import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl

import urllib.request
import zipfile
import glob
import pickle
from xml.dom import minidom


def get_label_data(tier, path):
    prep_object = DataPrep(tier, path)
    prep_object.load_label_key()
    prep_object.create_flipped_labelkey()
    return prep_object


# CWE data for data preparation
class CweFinder:
    def __init__(self, datapath):
        self.data_path = datapath
        self.cwefile = glob.glob(f'{self.data_path}/cwec*')
        self.download_cwe_file()
        self.weaknesses = minidom.parse(self.cwefile[0]).getElementsByTagName('Weakness')

    def download_cwe_file(self, force=False):
        if self.cwefile and not force:
            return
        path = f'{self.data_path}/cwes.zip'
        urllib.request.urlretrieve('https://cwe.mitre.org/data/xml/cwec_latest.xml.zip', path)

        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(self.data_path)
        zip_ref.close()
        self.cwefile = glob.glob(f'{self.data_path}/cwec*')

    def find_root_cwe(self, cwe_id):
        for elem in self.weaknesses:
            if elem.getAttribute('ID') == cwe_id:
                items = elem.getElementsByTagName('Related_Weakness')
                for item in items:
                    if item.getAttribute('Nature') == "ChildOf":
                        return item.getAttribute('CWE_ID')

    def find_cwe_children(self, cwe_id):
        children = []
        for elem in self.weaknesses:
            if elem.getAttribute('ID') == cwe_id:
                items = elem.getElementsByTagName('Related_Weakness')
                for item in items:
                    if item.getAttribute('Nature') == "ParentOf":
                        children.append(item.getAttribute('CWE_ID'))
                return children


# Data prep class
class DataPrep:

    def __init__(self, tier, data_path='/workspace/swagger_server/data'):
        self.data_path = data_path
        self.data_column = "text"
        self.class_column = "labels"
        self.label_key = dict()
        self.int_labels = dict()
        self.tier = tier

    def load_label_key(self):
        with open(f'{self.data_path}/t_{self.tier}_label_key.pickle', 'rb') as labels:
            self.label_key = pickle.load(labels)

    def create_flipped_labelkey(self):
        self.int_labels = dict((v,k) for k,v in self.label_key.items())


class CWEInferenceModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr = config.learning_rate
        self.model = config.base_model
        self.model.eval()
        self.classifier = nn.Linear(self.model.config.hidden_size, config.classes)
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor(config.class_weights))
        self.accuracy = torchmetrics.Accuracy()
        self.f1 = torchmetrics.F1Score(num_classes=config.classes)
        self.mcc = torchmetrics.MatthewsCorrCoef(num_classes=config.classes)
        self.save_hyperparameters()
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.base_output, config.base_output),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.base_output, config.classes))

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask, output_attentions=True)
        pooled_output = torch.mean(output.last_hidden_state, 1)
        return self.classifier(pooled_output), output.attentions


