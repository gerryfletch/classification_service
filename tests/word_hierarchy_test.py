import pytest
from classification.imagenet import Label
from classification.net import Classification
from classification.word_hierarchy import Hierarchy, Node
from classification.word_hierarchy import terrier_node, dog_node, animal_node
from nltk.corpus import wordnet


def test_node_from_word():
    node = Node.from_word('dog', 0, None)
    assert node.synset == wordnet.synset('dog.n.01')


def test_node_from_qualified_synset():
    node = Node.from_qualified_synset('dog.n.01', 0, None)
    assert node.synset == wordnet.synset('dog.n.01')


def test_node_from_synset():
    syn = wordnet.synset('dog.n.01')
    node = Node.from_synset(syn, 0, None)
    assert node.synset == syn


def test_terrier_groups_terrier():
    # 80% accuracy on 70% hierarchy should group in 'terrier'
    label = Label('02096294-n', 'australian terrier', '')
    c = Classification(0.8, label)
    node = Hierarchy(0.7).place([c])

    assert node == terrier_node


def test_terrier_groups_terrier_combination():
    # multiple classifications of terrier breeds should sum
    # to be 80% accuracy, result should be 'terrier'
    c1 = Classification(0.4, Label('02096294-n', 'australian terrier', ''))
    c2 = Classification(0.35, Label('02097658-n', 'silky terrier', ''))
    c3 = Classification(0.05, Label('02094433-n', 'yorkshire terrier', ''))

    node = Hierarchy(0.7).place([c1, c2, c3])

    assert node == terrier_node


def test_terrier_reduced_to_dog():
    c1 = Classification(0.4, Label('02096294-n', 'australian terrier', ''))
    # toy terrier is not actually a terrier breed, so does not exist as a 'terrier'
    c2 = Classification(0.4, Label('02087046-n', 'toy terrier', ''))

    node = Hierarchy(0.7).place([c1, c2])

    assert node == dog_node


def test_cat_and_dog_reduced_to_animal():
    c1 = Classification(0.4, Label('02096294-n', 'australian terrier', ''))
    c2 = Classification(0.4, Label('02123045-n', 'tabby, tabby cat', ''))

    node = Hierarchy(0.7).place([c1, c2])

    assert node == animal_node


def test_high_accuracy_hierarchy():
    c = Classification(0.85, Label('02096294-n', 'australian terrier', ''))

    node = Hierarchy(0.9).place([c])

    assert node is None


def test_non_matching_group():
    c = Classification(0.8, Label('04465501-n', 'tractor', ''))

    node = Hierarchy(0.7).place([c])

    assert node is None
