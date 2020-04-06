import pytest
from datasource import DataSource

valid_image_url = "https://www.sciencemag.org/sites/default/files/styles/article_main_large/public/dogs_1280p_0.jpg"


def test_invalid_url():
    url = "httpz://bla.xy wr"
    with pytest.raises(ValueError) as value_error:
        DataSource(url)
    assert str(value_error.value).startswith(
        f"Failed to read URL of data source.\nURL in question: {url}"
    )


def test_non_image_url():
    url = "http://google.com"
    with pytest.raises(ValueError) as value_error:
        DataSource(url)
    assert str(value_error.value).startswith(
        f"Failed to read data into Pillow image. URL in question: {url}"
    )


def test_tensor_is_not_none():
    ds = DataSource(valid_image_url)
    assert ds.tensor is not None
    ds.close()


def test_batch_shape():
    ds = DataSource(valid_image_url)
    assert ds.batch_tensor.shape == ds.tensor.unsqueeze(0).shape
    ds.close()
