import pytest
from nltk.corpus import wordnet
from classification import imagenet
from classification.imagenet import Label


def test_label_correct():
    label = imagenet.get_labels()[0]
    assert label.id == '01440764-n'
    assert label.name == 'tench, Tinca tinca'
    assert label.uri == 'http://wordnet-rdf.princeton.edu/wn30/01440764-n'

def test_cat_is_animal():
    cat = Label('02123045-n', 'tabby, tabby cat', '')
    animal = wordnet.synsets('animal')[0]

    assert cat.is_a(animal)

def test_dog_is_cat():
    dog = Label('02099601-n', 'golden retriever', '')
    cat = wordnet.of2ss('02123045-n')

    assert not dog.is_a(cat)

def test_piano_is_instrument():
    piano = Label('04515003-n', 'upright, upright piano', '')
    instrument = wordnet.synsets('musical_instrument')[0]

    assert piano.is_a(instrument)