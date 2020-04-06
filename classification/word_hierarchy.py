import nltk
from nltk.corpus import wordnet
from nltk.corpus import Synset

class Hierarchy:

  def __init__(self):
    nltk.download('wordnet')

  def get_synset(id: str) -> Synset:
    return wordnet.of2ss(id)


hierarchy = Hierarchy()
s = hierarchy.get_synset('02037110-n')
print(s)