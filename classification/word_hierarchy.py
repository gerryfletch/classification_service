import nltk
from nltk.corpus import wordnet
from classification.datasource import DataSource
from classification.imagenet import Label
from classification.net import Classification
from typing import Optional


class Node:
    def __init__(self, word: str, depth: int, parent, children: [] = []):
        self.synset = wordnet.synsets(word)[0]
        self.depth = depth
        self.parent = parent
        self.children = children
        self.count = 0

    def __str__(self):
        return f"Synset: {self.synset}, Depth: {self.depth}, Count: {self.count}"


# Tree Construction
root_node = Node("entity", 0, None)

# depth 1
animal_node = Node("animal", 1, root_node)
instrument_node = Node("instrument", 1, root_node)
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


class Hierarchy:

    def __init__(self):
        self.root = root_node

    def group_classification(self, classifications: [Classification]):
        curated_set = self.curate_initial_set(classifications)
        if curated_set == {}:
            return None

        # continuously reduce the curated set until an appropriate category
        # is found, and if it is never found, return None
        while self.select_highest(curated_set) is None:
            curated_set = self.reduce(curated_set)
            if curated_set is None:
                return None

        top_node = self.select_highest(curated_set)
        if top_node is not None:
            top_node.count = top_node.count + 1
        return top_node

    def curate_initial_set(self, classifications: [Classification]) -> dict:
        curated_set = {}
        for classification in classifications:
            closest_node = self.find_closest(classification, self.root)
            if closest_node == None or closest_node == self.root:
                continue
            if closest_node in curated_set:
                curated_set[closest_node] = curated_set[closest_node] + \
                    classification.accuracy
            else:
                curated_set[closest_node] = classification.accuracy
            print(
                f"{classification.label.name} fits group {closest_node} with total accuracy: {curated_set[closest_node]}")
        return curated_set

    def find_closest(self, c: Classification, node: Node) -> Node:
        for child_node in node.children:
            if c.label.is_a(child_node.synset):
                return self.find_closest(c, child_node)
        return node

    def reduce(self, curated_set: dict):
        max_depth = max(map(lambda n: n.depth, curated_set))
        if max_depth == 1:
            return None
        deep_nodes = list(filter(lambda n: n.depth == max_depth, curated_set))
        for node in deep_nodes:
            parent = node.parent
            if parent in curated_set:
                curated_set[parent] = curated_set[parent] + curated_set[node]
            else:
                curated_set[parent] = curated_set[node]
            del curated_set[node]

    def select_highest(self, curated_set: dict) -> Optional[Node]:
        for node in curated_set:
            if curated_set[node] >= 0.70:
                return node

    def print(self):
        self._print(self.root)

    def _print(self, node: Node):
        if node is None:
            return
        print(node)
        for child in node.children:
            self._print(child)
