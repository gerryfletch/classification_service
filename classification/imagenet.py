import torchvision.transforms as transforms
import functools

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


@functools.lru_cache(maxsize=1, typed=False)
def get_labels() -> [str]:
    '''
    Loads the ordered list of 1000 labels from imagenet_classes.txt
    '''
    with open('imagenet_classes.txt') as f:
        return [line.strip().lower() for line in f.readlines()]
