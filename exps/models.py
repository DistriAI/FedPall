from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
import torch 
import torch.utils.model_zoo as model_zoo

class Discriminator(nn.Module):
    # discriminator for adversarial training in ADCOL
    def __init__(self, base_model, client_num):
        super(Discriminator, self).__init__()
        try:
            in_features = base_model.classifier.fc4.in_features
        except:
            raise ValueError("base model has no classifier")
        self.discriminator = nn.Sequential(
            nn.Linear(in_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, client_num, bias=False),
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x

class adcol_model(nn.Module):
    def __init__(self, num_classes=10): 
        super(adcol_model, self).__init__()
        self.features = models.resnet50(weights=None)
        self.features.fc = nn.Identity()
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc4', nn.Linear(2048, 512)),
                ('bn4', nn.BatchNorm1d(512)),
                ('relu4', nn.ReLU(inplace=True)),
                ('fc5', nn.Linear(512, 512)),
                ('bn5', nn.BatchNorm1d(512)),
                ('relu5', nn.ReLU(inplace=True)),
                ('fc6', nn.Linear(512, num_classes)),
            ])
        )
    
    def forward(self, x):
        x1 = self.features(x)
        x = self.classifier(x1)
        return x1, x

class classifier_model(nn.Module):
    # discriminator for adversarial training in ADCOL
    def __init__(self, args, num_classes):
        super(classifier_model, self).__init__()     
        # 分类层
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc4', nn.Linear(2048, 512)),
                ('bn4', nn.BatchNorm1d(512)),
                ('relu4', nn.ReLU(inplace=True)),
                ('fc5', nn.Linear(512, 512)),
                ('bn5', nn.BatchNorm1d(512)),
                ('relu5', nn.ReLU(inplace=True)),
                ('fc6', nn.Linear(512, num_classes)),
            ])
        )
        
    def forward(self, x):
        x = self.classifier(x)
        return x

class GateModel(nn.Module):
    # discriminator for adversarial training in ADCOL
    def __init__(self, args, client_num):
        super(GateModel, self).__init__()     
        # 分类层
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc4', nn.Linear(2048, 512)),
                ('bn4', nn.BatchNorm1d(512)),
                ('relu4', nn.ReLU(inplace=True)),
                ('fc5', nn.Linear(512, 512)),
                ('bn5', nn.BatchNorm1d(512)),
                ('relu5', nn.ReLU(inplace=True)),
                ('fc6', nn.Linear(512, client_num)),
            ])
        )
        
    def forward(self, x):
        x = self.classifier(x)
        return x



