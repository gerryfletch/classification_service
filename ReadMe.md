# Image Classification
This service aims to classify images using the [ImageNet](http://www.image-net.org/) database,
which labels images based on the [WordNet](https://wordnet.princeton.edu/) noun hierarchy.

Image classifications are matched to nodes in the [Curated Tree](curated-tree) if their
accumulated accuracy is greater than the threshold (default to 0.7). Details on this process
are described in the [Matching Algorithm](matching-algorithm) section.

The in-memory curated tree can be queried for insights such as the
total number of classifications belonging to dogs, for example.

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
```

## Matching Algorithm
The [ImageNet](http://www.image-net.org/) database labels images against one-thousand labels
from the [WordNet](https://wordnet.princeton.edu/) noun hierarchy.

The [Neural Network API](./classification/net.py) will default to an accuracy tolerance of 5%. Any classifications below this accuracy are discarded. 

The [Word Hierarchy API](./classification/word_hierarchy.py) accepts a list of classifications.

### 1. Closest-Node Grouping
Each classification is grouped with the closest node in the curated tree based on the labels
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
with their parent. If the neural network was 20% certain that the image was a `dog`
(based on generalisation of some lower breed, e.g dalmation), and 55% certain that
the image was a form of terrier, the tree would look like this after the initial
closest-match phase:

```
entity
  - animals
    - dogs (accuracy=0.20)
      - terriers (accuracy=0.55)
      - retrievers
    - cats
  - instruments
    - drum
    - piano
    - guitar
```

The reduce step identifies `terriers` as the lowest node, and combines its accuracy
result with its parent, `dogs`. The terrier accuracy is then discarded, forming
the following tree:
```
entity
  - animals
    - dogs (accuracy=0.75)
      - terriers
      - retrievers
    - cats
  - instruments
    - drum
    - piano
    - guitar
```

An accuracy greater than the threshold has now been met, so the classification result
is `dogs`. This is an overly simplistic example, but this algorithm is designed to
work for both small graphs such as this example and highly distributed trees
where there may be many classifications with low accuracies. 

## Usage
-

## Output
-