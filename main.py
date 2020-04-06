from classification.datasource import DataSource
from classification.net import Network

urls = [
    "https://homepages.cae.wisc.edu/~ece533/images/fruits.png"
]

net = Network()

for i in range(len(urls)):
    ds = DataSource(urls[i])
    print(net.classify(ds))
