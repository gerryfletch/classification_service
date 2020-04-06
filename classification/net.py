import torchvision.models as models
import torch
from torch import Tensor
import torch.nn.functional as F
from classification.datasource import DataSource
from classification import imagenet


class Classification:
    accuracy: int
    label: str

    def __init__(self, accuracy: int, label: int):
        self.accuracy = accuracy
        self.label = label


class Network():
    def __init__(self, accuracy_boundary: int = 0.6, use_cuda: bool = False):
        self.accuracy_boundary = accuracy_boundary
        self.use_cuda = use_cuda
        model = models.resnet152(pretrained=True)
        if self.use_cuda:
            model.to('cuda')

        model.eval()
        self.model = model
        self.labels = imagenet.get_labels()

    def classify(self, data_source: DataSource) -> [Classification]:
        '''
        Classifies the image against the model, and closes the datasource.
        The datasource may not be reclassified after this operation.
        '''
        if self.use_cuda:
            data_source.batch_tensor.to('cuda')
        with torch.no_grad():
            result = self.model(data_source.batch_tensor)
        data_source.close()
        normalised_result = F.softmax(result[0], dim=0)
        return self._refine_output(normalised_result)

    def classify_url(self, url: str) -> [(int, str)]:
        '''
        Classifies the image stored in the URL against the model.
        Utility method delagating to classify with a built datasource.
        '''
        return self.classify(DataSource(url))

    def _refine_output(self, result: Tensor) -> [Classification]:
        '''
        Takes the top 5 results and filters them by an accuracy of >= 60%
        '''
        top_values, top_indexes = torch.topk(result, 5)
        top_result = []
        for i in range(5):
            print(self.labels[top_indexes[i]])
            if (top_values[i] >= self.accuracy_boundary):
                top_result.append(
                    Classification(top_values[i], self.labels[top_indexes[i]])
                )
        return top_result
