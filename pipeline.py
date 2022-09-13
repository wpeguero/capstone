"""Algorithms used to process data before modeling.

...

A set of algorithms used to feed in and process
data before used within the model. This will contain
the data extraction from its rawest form and output
the final form of the data set. The main source of
data will be image related from the Cancer Imaging
Archive.
"""
import requests
import pandas as pd
import json
from os.path import exists
import os
import platform
from pathlib import Path
import tensorflow as tf
import pydicom

def main():
    """Test the new functions."""
    data, val_ds = load_data('data/brain_tumor_dataset/')


def get_data(name:str, option:str) -> list: # This will be deprecated and no longer in use
    """Extract metadata using NBIA api.

    ...

    Uses the requests module to get the metadata
    using the Cancer Imaging Archive's NBIA api
    to extract the metadata into a pandas DataFrame.

    Parameter(s)
    ---

    name:str
        The name of the research data set.

    option:str
        The kind of api call to be made. the following
        are the possible calls that one can make:
            1. collections - Get a list of collections in the current IDC data version.
            2. cohorts - Get the metadata on the user's cohorts.

    Returns
    ---
    key_data:list
        A list of the samples within the data set.
    """
    assert option is not None, "Please select between on of the following two options:\n1. collections\n2. cohorts\n\nFor more information, please view documentation."
    base_link = "https://api.imaging.datacommons.cancer.gov/v1/"
    if option == "collections":
        full_link = base_link + option
    elif option == "cohorts":
        full_link = base_link + option
    else:
        pass
    response = requests.get(full_link)
    assert response.status_code == 200, "Authorization Error: {}".format(response.status_code)
    key_data = response.json()
    if exists('keys.txt') is False:
        with open("keys.txt", 'w') as fp:
            fp.write(str(key_data))
            fp.close()
    else:
        pass
    return key_data

def obtain_data(filename:str):
    """Extract the data from the .dcm files.
    
    ...

    Loads the data using the pydicom library to extract both metadata and more.
    """
    pass

def load_data(directory:str):
    """Load the data using tensorflow data set library.

    ...

    Uses the os library and the TensorFlow Data
    api to load, batch, and process the data for
    training.
    ---

    Parameter(s)

    directory:str
        path to the folder containing all of the images within the correct order.
    """
    data_dir = Path(directory)
    image_count = len(list(data_dir.glob(f"*/*")))
    BATCH_SIZE = 32
    BUFFER_SIZE = image_count
    img_height = 180
    img_width = 180
    train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=BATCH_SIZE
            )
    val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=BATCH_SIZE
            )
    dataset = train_ds.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=BUFFER_SIZE)
    return dataset, val_ds

def extract_data(url_file:str) -> None:
    """Extract the images using its url address.

    ...

    Uses the manifest file obtained from the
    Imaging Data Commons (IDC) to extract the images.
    This is mainly using the os library and its
    ability to throw shell commands.
    """
    if platform.system() is "Linux":
        os.system("cat manifest.txt | gsutil -m cp -I")
    elif platform.system() is "Windows":
        os.system("type manifest.txt | gsutil -m cp -I")


if __name__ == "__main__":
    main()
