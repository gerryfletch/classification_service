import torchvision.models as models
import torch
from torch import Tensor
import torch.nn.functional as F
from classification.datasource import DataSource
from classification import imagenet
from classification.imagenet import Label


class Classification:
    accuracy: float
    label: Label

    def __init__(self, accuracy: float, label: Label):
        self.accuracy = accuracy
        self.label = label


class Network():
    def __init__(self, accuracy_boundary: float = 0.6, use_cuda: bool = False):
        self.accuracy_boundary = accuracy_boundary
        self.use_cuda = use_cuda
        model = models.resnet152(pretrained=True)
        if self.use_cuda:
            model.to('cuda')

        model.eval()
        self.model = model
        self.labels = imagenet.get_labels()

    def classify_url(self, url: str) -> [Classification]:
        '''
        Classifies the image stored in the URL against the model.
        Utility method delagating to classify with a built datasource.
        '''
        return self.classify(DataSource(url))

    def classify(self, data_source: DataSource) -> [Classification]:
        '''
        Classifies the image against the model, and closes the datasource.
        The datasource may not be reclassified after this operation.
        All returned classifications have an accuracy greater than zero.
        '''
        if self.use_cuda:
            data_source.batch_tensor.to('cuda')
        with torch.no_grad():
            result = self.model(data_source.batch_tensor)

        data_source.close()
        normalised_result = F.softmax(result[0], dim=0)
        return self._convert_tensor_to_classification(normalised_result)

    def _convert_tensor_to_classification(self, tensor: Tensor) -> [Classification]:
        '''
        Converts all results with more than five percent (5%) accuracy to classifications.
        '''
        values, indexes = torch.topk(tensor, len(tensor))
        classifications = []
        for i in range(len(tensor)):
            if values[i] < 0.05:
                continue
            classification = Classification(
                values[i], self.labels[indexes[i]]
            )
            classifications.append(classification)
        return classifications
