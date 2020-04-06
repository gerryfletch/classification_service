import pytest
from classification import imagenet

def test_classes_lower_case():
  assert all(label.islower() for label in imagenet.get_labels())