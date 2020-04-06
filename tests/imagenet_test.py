import pytest
from classification import imagenet


def test_label_correct():
  label = imagenet.get_labels()[0]
  assert label.id == '01440764-n'
  assert label.name == 'tench, Tinca tinca'
  assert label.uri == 'http://wordnet-rdf.princeton.edu/wn30/01440764-n'