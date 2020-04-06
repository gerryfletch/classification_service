import os
import uuid
import urllib.request
from PIL import Image
import torchvision.transforms as transforms
import imagenet
from torch.tensor import Tensor


def _delete_file(file_name: str):
    '''Removes the image from the local file system.'''
    os.remove(file_name)


def _attempt_image_load(url: str, file_name: str) -> Image:
    '''Loads the image onto disk returning an Image object'''
    try:
        urllib.request.urlretrieve(url, file_name)
    except Exception as exception:
        raise ValueError(
            f"Failed to read URL of data source.\nURL in question: {url}\nMessage: {str(exception)}"
        )
    try:
        return Image.open(file_name)
    except:
        _delete_file(file_name)
        raise ValueError(
            f"Failed to read data into Pillow image. URL in question: {url}"
        )


class DataSource:
    '''
    Utility class to manage file:
      - IO
      - Normalization
      - Resize + Crop
      - Deletion
    '''

    url: str
    file_name: str
    raw_image: Image
    tensor: Tensor
    batch_tensor: Tensor

    def __init__(self, url: str):
        '''Loads the datasource from a URL.'''
        self.url = url
        self.file_name = "a-" + str(uuid.uuid4())
        self.raw_image = _attempt_image_load(self.url, self.file_name)
        self.tensor = self._process_image_to_tensor(self.raw_image)
        self.batch_tensor = self.tensor.unsqueeze(0)

    def _process_image_to_tensor(self, image: Image) -> Tensor:
        processor = transforms.Compose([
            transforms.Resize(imagenet.image_edge_length),
            transforms.CenterCrop(imagenet.crop_size),
            transforms.ToTensor(),
            imagenet.normalize_transform
        ])

        return processor(image)

    def close(self):
        '''Closes Pillow connection and removes the image from the local file system.'''
        self.raw_image.close()
        _delete_file(self.file_name)
