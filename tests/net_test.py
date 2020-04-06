import pytest
from classification.datasource import DataSource
from classification.net import Network


net = Network(accuracy_boundary=0.3)
golden_retriever_url = "https://post.healthline.com/wp-content/uploads/sites/3/2020/02/322868_1100-1100x628.jpg"


def test_sanity_recognition_with_datasource():
    ds = DataSource(golden_retriever_url)
    assert net.classify(ds)[0].label == 'golden retriever'

def test_sanity_recognition_with_url():
    assert net.classify_url(golden_retriever_url)[0].label == 'golden retriever'

def test_sanity_recognition_accuracy():
    pass
