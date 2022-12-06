"""Test models and model library for capstone project."""
from models import *
from pipeline import load_testing_data, predict, calculate_confusion_matrix
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import CategoricalCrossentropy
import pytest

def test_model_compilation():
    """Test whether the model compiles successfully or not."""
    inputs, output = tumor_classifier(1147,957)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='Adam', loss=CategoricalCrossentropy())

pytest.fixture
def create_test_data():
    """Loads the data for testing purposes."""
    filename = "data/CMMD-set/test.csv"
    tdata = load_testing_data(filename, sample_size=1_000)
    return tdata

@pytest.fixture
def test_model_output(create_test_data):
    """Test whether the output is recorded within an array."""
    inputs, output = tumor_classifier(1147,957)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='Adam', loss=CategoricalCrossentropy())
    pdf = model.predict(create_test_data[0])
    assert type(pdf['score'].values[0]) == list()

@pytest.fixture
def test_model_score_probability(create_test_data):
    """Test whether the output of the models are a set of probabilities."""
    inputs, output = tumor_classifier(1147,957)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='Adam', loss=CategoricalCrossentropy())
    pdf = model.predict(create_test_data[0])
    assert sum(pdf['score'].values[0]) == 1.0

@pytest.fixture
def test_model_accuracy(create_test_data):
    """Test whether the accuracy of the most recent model reaches the standard."""
    pdf = predict(create_test_data, 'models/tclass_VGG8')
    ct, metrics = calculate_confusion_matrix(pdf)
    assert metrics['Accuracy'] >= 0.90

if __name__ == "__main__":
    pytest.main()
