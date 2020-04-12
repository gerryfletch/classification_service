import nltk
from nltk.corpus import wordnet
from classification.datasource import DataSource
from classification.imagenet import Label
from classification.net import Classification


dog = wordnet.synsets("dog")[0]
terrier = wordnet.synsets("terrier")[0]
retriever = wordnet.synsets("retriever")[0]

cat = wordnet.synsets("cat")[0]


australian_terrier = wordnet.of2ss('02096294-n')
silky_terrier = wordnet.of2ss('02097658-n')
yorkshire_terrier = wordnet.of2ss('02094433-n')

labrador_retriever = wordnet.of2ss('02099712-n')

egyptian_cat = wordnet.of2ss('02124075-n')

animal = wordnet.synsets("animal")[0]

piano = wordnet.synsets("piano")[0]
guitar = wordnet.synsets("guitar")[0]
drum = wordnet.synsets("drum")[0]
instrument = wordnet.synsets("instrument")[0]

religious_person = wordnet.synsets("religion")[0]


class Node:
  def __init__(self, synset, children: []):
    self.synset = synset
    self.children = children
    self.count = 0


cat_node = Node(cat, [])

terrier_node = Node(terrier, [])
retriever_node = Node(retriever, [])

dog_node = Node(dog, [terrier_node, retriever_node])

animal_node = Node(animal, [dog_node, cat_node])

piano_node = Node(piano, [])
guitar_node = Node(guitar, [])
drum_node = Node(drum, [])
instrument_node = Node(instrument, [piano_node, guitar_node, drum_node])

class_hierarchy = Node(None, [animal_node, instrument_node])

class Place:
  def __init__(self, node: Node, similarity: float):
    self.node = node
    self.similarity = similarity

def go(synset):
  p = place(synset, class_hierarchy, 0)
  if p.similarity >= 0.5:
    p.node.count = p.node.count + 1
  return p

def place(synset, root, best_similarity):
  best_node = root
  for node in root.children:
    similarity = calc_sim(synset, node.synset)
    if similarity > best_similarity:
      best_similarity = similarity
      best_node = node
  if best_node == root:
    print(best_similarity)
    return Place(best_node, best_similarity)
  else:
    return place(synset, best_node, best_similarity)


def calc_sim(s1, s2):
  # s1.wup_similarity(s2)
  # return s1.path_similarity(s2)
  return s1.lch_similarity(s2)


class Group:
  def __init__(self, node: Node, similarity: float):
    self.node = node
    self.similarity = similarity


class Hierarchy:

    def __init__(self):
        nltk.download('wordnet')

    def get_synset(self, id: str):
        return wordnet.of2ss(id)

    def add_classification(self, classification: Classification) -> Group:
      synset = self.get_synset(classification.label.id)
      hypernyms = synset.hypernyms
      group = self.identify_group(synset, class_hierarchy, 0)
      if group.similarity > 0.5:
        print(f"Placing classification {synset} in group {group.node.synset}.")
        group.node.count = group.node.count + 1
      return group

    # def identify_is_a_group(self, hypernyms, root: Node) -> Node:
    #   if root.synset in hypernyms:
    #     return root
    #   for node in root.children:

    def identify_group(self, synset, root: Node, best_similarity: int) -> Group:
      best_node = root
      for node in root.children:
        similarity = self.calc_similarity(synset, node.synset)
        if similarity > best_similarity:
          best_similarity = similarity
          best_node = node
      if best_node == root:
        return Group(best_node, best_similarity)
      else:
        return self.identify_group(synset, best_node, best_similarity)

    def calc_similarity(self, s1, s2):
      return s1.lch_similarity(s2)
