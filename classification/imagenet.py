import torchvision.transforms as transforms
import functools
from classification import labels
from typing import Dict

# Normalize function for mean + standard deviation
# used for all imagenet images.
# Transforms the array into a normalized RGB array
normalize_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# Used to compress the image to a max of 256 edge pixels
image_edge_length = 256


# Slightly smaller crop of the compressed image for processing
crop_size = 224


class Label:
    id: str
    name: str
    uri: str

    def __init__(self, id: str, name: str, uri: str):
        self.id = id
        self.name = name
        self.uri = uri


@functools.lru_cache(maxsize=1, typed=False)
def get_labels() -> [Label]:
    result = []
    for i in range(1000):
        label_dic = labels.labels.get(i)
        label = Label(
            id=label_dic['id'],
            name=label_dic['label'],
            uri=label_dic['uri']
        )
        result.append(label)
    return result