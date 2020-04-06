from classification.datasource import DataSource
from classification.net import Network

urls = [
    "https://homepages.cae.wisc.edu/~ece533/images/fruits.png"
]

net = Network(accuracy_boundary=0.2)

for i in range(len(urls)):
    ds = DataSource(urls[i])
    print(net.classify(ds))