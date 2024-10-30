from data_processor.statistics import mean, median

def test_mean():
    assert mean([1, 2, 3, 4, 5]) == 3.0

def test_median():
    assert median([1, 2, 3, 4, 5]) == 3.0
