from classification.datasource import DataSource
from classification.net import Network
from classification.net import Classification
from classification import word_hierarchy
from classification.word_hierarchy import Hierarchy
from nltk.corpus import wordnet

urls = [
    # "https://i.pinimg.com/originals/99/f9/ed/99f9ede31328c8484e9e252d08811535.jpg",
    # "https://vetstreet.brightspotcdn.com/dims4/default/851240c/2147483647/thumbnail/645x380/quality/90/?url=https%3A%2F%2Fvetstreet-brightspot.s3.amazonaws.com%2Fdd%2F5c%2Fb14d4033429eaf1ebcd52f8a9d10%2Fyorkshire-terrier-ap-0mhxtf-645.jpg",
    "https://c8.alamy.com/comp/AJNJT5/labrador-retriever-and-yorkshire-terrier-AJNJT5.jpg",
    # "https://www.cats.org.uk/media/3236/choosing-a-cat.jpg",
    # "https://st.mascus.com/imagetilewm/product/mclean/kioti-dk551c-tractor-st5573,141778_1.jpg"
    # "https://lh3.googleusercontent.com/proxy/b1FOSTRZ0eEtIm1D74PlX7JCBzhcMBw5AdPXYd__OX6FYP7i13Hm6P-aNbTXVpfXC_QvpbUNCsresVNd--rznl_vZxossqgTozFUCPU6pMwUi3rwDaZy60yT-Np6uvD1K_X6h7zVqljvtPB3ydL7ZjQINmGrBFq6-u45mXbYXvo671NlZLTddUQj8CfGJqRm"
    # "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&w=1000&q=80"
]

# print(wordnet.of2ss('02096294-n').hypernyms)
# print(wordnet.synsets("dog"))

net = Network(accuracy_boundary=0)
# h = Hierarchy()

# for url in urls:
#     classification = net.classify_url(url)[0]
#     group = h.add_classification(classification)
#     print(group.similarity)



for i in range(len(urls)):
    ds = DataSource(urls[i])
    classifications = net.classify(ds)
    print(len(classifications))
    n = wordnet.of2ss(classifications[0].label.id)

    for i in range(len(classifications)):
        classification = classifications[i]
        if classification.accuracy * 100 == 0:
            print("classification accuracy is zero")
        print('Accuracy of %5s : %2d %%. Id: %s' % (
            classification.label.name,
            classification.accuracy * 100,
            classification.label.id
        ))

    # for classification in net.classify(ds):
    #     print('\n\n')
    #     print('========================')
    #     print('Accuracy of %5s : %2d %%' % (
    #         classification.label.name,
    #         classification.accuracy * 100
    #     ))
    #     synset = h.get_synset(classification.label.id)
    #     while synset.hypernyms():
    #         print(synset)
    #         synset = synset.hypernyms()[0]
