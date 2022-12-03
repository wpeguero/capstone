"""Test models and model library for capstone project."""
from models import *
from pipeline import load_testing_data, predict
from tensorflow.keras.models import Model
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
    tdata = load_testing_data(filename, sample_size=1)
    return tdata

def test_model_output(create_model):
    """Test whether the output of the models are a set of probabilities."""
    inputs, output = tumor_classifier(1147,957)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='Adam', loss=CategoricalCrossentropy())
    prediction = model.predict(create_model[0])


if __name__ == "__main__":
    #pytest.main()
    inputs, output = tumor_classifier(1147,957)
    model_name = "models/tclass_VGG7"
    tdata = create_test_data()
    pdf = predict(tdata, model_name)
    print(pdf['score'].tolist())
    print(type(pdf['score'].tolist()))
