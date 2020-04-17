# Image Classification
This service aims to classify images using the [ImageNet](http://www.image-net.org/) database,
which labels images based on the [WordNet](https://wordnet.princeton.edu/) noun hierarchy.

Image classifications are matched to nodes in the [Curated Tree](#curated-tree) if their
accumulated accuracy is greater than the threshold (default to 0.7). Details on this process
are described in the [Matching Algorithm](#matching-algorithm) section.

The in-memory curated tree can be queried for insights such as the
total number of classifications belonging to dogs, for example.

# Contents
- [Install](#install)
- [Usage](#usage)
- [Curated Tree](#curated-tree)
- [Matching Algorithm](#matching-algorithm)
  - [Step 1: Closest-Node Grouping](#1.-closest-node-grouping)
  - [Step 2: Reduction](#2.-reduction)
  - [Conclusion](#Conclusion)

## Install
Requires Python 3.7+

```bash
git clone https://github.com/gerryfletch/classification_service/
cd classification_service
conda install --yes --channel pytorch --file requirements.txt

# run tests to make sure everything is working
pytest tests/

# run the demo data + visualize results
python3.7 main.py
```

## Usage
The application is split into three modules.
- Neural network creation and configuration is under `classification/net.py`
- Data sources and image manipulation is under `classification/datasource.py`
- Hierarchal grouping with the curated tree is under `classification/word_hierarchy.py`

### Construct a Data Source, Classify, and Store in Hierarchy
```python
data_source = DataSource(url = "URL_TO_IMAGE")
net = Network(accuracy_boundary=0.05, use_cuda=False) # boundary between 0-1, defaults to 0.05. use_cuda defaults to False.

classifications = net.classify(data_source) # : [Classification]
# or, net.classify_url("URL_TO_IMAGE") for shorthand

hierarchy = Hierarchy(accuracy_threshold=0.7) # threshold between 0-1, defaults to 0.7.

hierarchy.place(classifications) # : bool

hierarchy.print() # visualizes results in tree
```

## Curated Tree
```
entity
  - animals
    - dogs
      - terriers
      - retrievers
    - cats
  - instruments
    - drum
    - piano
    - guitar
  - food
```

## Matching Algorithm
The [ImageNet](http://www.image-net.org/) database labels images against one-thousand labels
from the [WordNet](https://wordnet.princeton.edu/) noun hierarchy.

The [Neural Network API](./classification/net.py) will default to an accuracy tolerance of 5%. Any classifications below this accuracy are discarded. 

The [Word Hierarchy API](./classification/word_hierarchy.py) accepts a list of classifications.

### 1. Closest-Node Grouping
Each classification is grouped with the closest node in the curated tree based on the label's
`is-a` relationship to the node's name. Specifically, this step recursively retrieves
the hypernyms of the label and checks for the existence of the node name. In the general sense, this allows
checks such as `cat is-a animal`.

The accuracy of classifications which result in the same node are combined. For example, two
separate fish breeds with accuracy `0.4` and `0.3` respectively may generalise to the Animal node
with accuracy `0.7`.

Classifications which are closest to the `entity` type are discarded, regardless of their accuracy.
For example, `tractor` is neither an `animal` or `instrument`, so the closest `is-a` relationship
is `entity`, and is therefor discarded.

An example of this step with the following simplistic accuracies:
```
Accuracy of Tabby Cat : 40 %
Accuracy of Australian terrier : 30 %
Accuracy of Yorkshire Terrier : 30 %
```

Would result in the following tree:
```
entity
  - animals
    - dogs
      - terriers | accuracy = 0.6 (0.3 + 0.3)
      - retrievers
    - cats | accuracy = 0.4
  - instruments
    - drum
    - piano
    - guitar
```

Both terrier breed classifications have been matched to the `terrier` node
based on their granular `is-a` relationship, and their accuracies combined.

### 2. Reduction
If there is no node with an accuracy greater than the threshold (default 0.70), the tree
is continuously reduced and combined until either:
- a node meets the threshold
- the tree is reduced to the `entity` (root) node

Reducing the tree works by selecting the deepest nodes, and combining their accuracy
with their parent. Continuing from the example in step one, the deepest node is
`terriers` with a depth of 3 (denoted by the left-side number in the display below).

```
0 entity
  1 - animals
    2 - dogs
      3 - terriers | accuracy = 0.6 (0.3 + 0.3)
      3 - retrievers
    2 - cats | accuracy = 0.4
  1 - instruments
    2 - drum
    2 - piano
    2 - guitar
```

The accuracies for the `terriers` node are combined with the accuracy of it's parent,
the `dogs` node, which has an implicit accuracy of zero. The tree now looks like this:

```
0 entity
  1 - animals
    2 - dogs | accuracy = 0.6 (0.3 + 0.3)
      3 - terriers
      3 - retrievers
    2 - cats | accuracy = 0.4
  1 - instruments
    2 - drum
    2 - piano
    2 - guitar
```

When the reduction phase is applied again, the deepest nodes are both `dogs` and `cats`.
The accuracies of each are lifted to their parent, which is shared by both nodes: `animals`.
Since the node is shared, the accuracies are combined, resulting in the following tree:

```
0 entity
  1 - animals | accuracy = 1.0 (0.4 + (0.3 + 0.3))
    2 - dogs
      3 - terriers
      3 - retrievers
    2 - cats
  1 - instruments
    2 - drum
    2 - piano
    2 - guitar
```

The result is a one hundred percent certainty that the classified image was of an `animal`.
Since the threshold (`>= 0.7`) has been met, the resulting node is returned to the client.

### Conclusion
The example given is overly simplistic and works within a very small universe of
possibilities. This algorithm is designed to work for highly distributed trees
with very granular results. The continuous reduction step is an alternative to
relying on the similarity (e.g Wu-Palmer) of nouns to accumulate accuracies and 
eventually pick a node, which only works if the universe of the neural network
results is enclosed in the curated tree.