import nltk
from nltk.corpus import wordnet
from classification.datasource import DataSource
from classification.imagenet import Label
from classification.net import Classification
from typing import Optional, Dict


class Node:
    def __init__(self, word: str, depth: int, parent, children: [] = []):
        self.synset = wordnet.synsets(word)[0]
        self.depth = depth
        self.parent = parent
        self.children = children
        self.count = 0

    def __str__(self):
        return f"Synset: {self.synset}, Depth: {self.depth}, Count: {self.count}"


# Doubly-Connected Tree Construction
root_node = Node("entity", 0, None)

# depth 1
animal_node = Node("animal", 1, root_node)
instrument_node = Node("musical_instrument", 1, root_node)
root_node.children = [animal_node, instrument_node]

# depth 2
cat_node = Node("cat", 2, animal_node)
dog_node = Node("dog", 2, animal_node)
animal_node.children = [cat_node, dog_node]

piano_node = Node("piano", 2, instrument_node)
guitar_node = Node("guitar", 2, instrument_node)
drum_node = Node("drum", 2, instrument_node)
instrument_node.children = [piano_node, guitar_node, drum_node]

# depth 3
terrier_node = Node("terrier", 3, dog_node)
retriever_node = Node("retriever", 3, dog_node)
dog_node.children = [terrier_node, retriever_node]


class Curation:
    '''
    Class to handle all the logic for grouping classifications
    against the curated tree. The algorithm is composed of
    reduce and combine operations.

    First, each classification is matched against a node based
    on is-a relationships into a curation dictionary.
    Classifications matching the root node are discarded, and
    nodes with multiple matches are combined, accumulating the
    classification accuracy.

    Then the client may reduce the curation until an accuracy
    threshold is met. The lowest (deepest) nodes in the
    curated tree are lifted by one layer, combining their
    accuracy with their parents.
    See Curation#reduce for a detailed example.
    '''

    def __init__(self, classifications: [Classification]):
        self.curated = self.curate_classifications(classifications)

    def curate_classifications(self, classifications: [Classification]) -> Dict[Node, float]:
        '''
        Finds the closest suitable node for a classification and keys it against the accuracy.
        If there is no suitable node, the classification is discarded.
        Classifications with the same node are combined.
        See self.combine
        '''
        curated = {}
        for classification in classifications:
            closest_node = self.find_closest_node(classification, root_node)
            # discard classification if the closest node is the root
            if closest_node == root_node:
                continue
            self.combine(curated, closest_node, classification.accuracy)
        return curated

    def find_closest_node(self, classification: Classification, node: Node) -> Node:
        '''
        Recursively visits nodes based on their is-a relationship.
        In a set of child nodes, logically only one may have an is-a relationship.
        If no children nodes hold this relationship, the current node is returned.
        '''
        for child in node.children:
            if classification.label.is_a(child.synset):
                return self.find_closest_node(classification, child)
        return node

    def get_node_above_accuracy(self, accuracy_threshold: float) -> Optional[Node]:
        '''
        Selects the first node with an accuracy greater than or equal to the threshold.
        '''
        for node, accuracy in self.curated.items():
            if accuracy >= accuracy_threshold:
                return node
        return None

    def combine(self, d: Dict[Node, float], n: Node, a: float):
        '''
        Sums the value at d[n] with a, storing the result in d[n].
        '''
        if n in d:
            d[n] = d[n] + a
        else:
            d[n] = a

    def reduce_until(self, threshold: float) -> Optional[Node]:
        '''
        Continuously reduces the classifications until a group has
        an accuracy greater than or equal to the treshold.
        If this threshold is never met, or the group is generalised
        to the object level, None is returned.
        '''
        while self.get_node_above_accuracy(threshold) == None:
            self.reduce()
            if self.curated == {}:
                # reduced to empty set
                return None
        return self.get_node_above_accuracy(threshold)

    def reduce(self):
        '''
        Lifts the lowest classifications in the curated graph by
        one level, and accumulates the accuracies of common
        classifications. For example,
        (name=Dog_A, depth=2, accuracy=0.30)
        (name=Dog_B, depth=2, accuracy=0.35)
        Would reduce to
        (name=Animal, depth=1, accuracy=0.30)
        (name=Animal, depth=1, accuracy=0.35)
        The accuracies are then accumulated, resulting in:
        (name=Animal, depth=1, accuracy=0.65)
        '''
        if self.curated == {}:
            return

        max_depth = max(map(lambda n: n.depth, self.curated))
        if max_depth == 1:
            self.curated = {}

        deep_nodes = list(filter(lambda n: n.depth == max_depth, self.curated))

        for child in deep_nodes:
            accuracy = self.curated[child]
            parent = child.parent
            self.combine(self.curated, parent, accuracy)
            del self.curated[child]


class Hierarchy:
    '''
    The Hierarchy class is responsible for maintaining the catalog
    of classification nodes and their respective metadata, providing
    query functions for placed classifications.
    '''

    accuracy_threshold: float
    hierarchy: Dict[Node, int]

    def __init__(self, accuracy_threshold: float):
        self.accuracy_threshold = accuracy_threshold
        self.hierarchy = {}

    def place(self, classifications: [Classification]) -> Optional[Node]:
        curation = Curation(classifications)
        curated_group = curation.reduce_until(self.accuracy_threshold)
        if curated_group is not None:
            print(f"{curated_group.synset.lemma_names()[0]}")
            self._store(curated_group)
        else:
            print("n/a - top n/3 categories:")
            for i in range(min(3, len(classifications))):
                print(f"{classifications[i].label.name} : {classifications[i].accuracy}")
        return curated_group

    def query_recursive_count(self, root: Node) -> int:
        total = self.hierarchy.get(root, 0)
        for child in root.children:
            total += self.query_recursive_count(child)
        return total

    def query_count(self, node: Node) -> int:
        return self.hierarchy.get(node, 0)

    def _store(self, node: Node):
        self.hierarchy.setdefault(node, 0)
        self.hierarchy[node] = self.hierarchy[node] + 1

    def print(self):
        self._print(root_node)

    def _print(self, node: Node):
        string = ""
        for i in range(node.depth):
            string += "  "
        count = self.query_count(node)
        name = node.synset.lemma_names()[0]
        string += f"{str(name)} - count: {count}"

        print(string)

        for child in node.children:
            self._print(child)
