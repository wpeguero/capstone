"""Algorithms used to process data before modeling.

...

A set of algorithms used to feed in and process
data before used within the model. This will contain
the data extraction from its rawest form and output
the final form of the data set. The main source of
data will be image related from the Cancer Imaging
Archive.
"""
import pandas as pd
import os
from pathlib import Path
import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt

def main():
    """Test the new functions."""
    filename = "data/MRI_Duke_Image_Data/MRIDID/Breast_MRI_001/ax dyn 1st pass/1-089.dcm"
    ds = pydicom.dcmread(filename)
    slices = ds.pixel_array
    sex = ds.PatientSex
    if sex == "F":
        sex = 1
    else:
        sex = 2
    age = ds.PatientAge
    age = age.replace("Y", "")
    age = int(age)
    weight = int(ds.PatientWeight)
    modality = ds.Modality
    if modality == 'MR':
        modality = 1
    elif modality == 'CT':
        modality = 2
    elif modality == 'PT':
        modality = 3
    else:
        modality = 4
    pregnancy_status = ds.PregnancyStatus
    PID = ds.PatientID
    if slices.ndim <= 2:
        pass
    elif slices.ndim >= 3:
        slices = slices[0]
    datapoint = {"Patient ID":PID, "Image":slices, "metadata": [sex, age, weight, modality, pregnancy_status]}
    print(slices.shape)


def _extract_key_images(new_download = False):
    """Extract the key images based on the Annotation Boxes file.
    
    Grabs the images from the full directory and
    moves them to a separate directory for keeping
    only the key data."""
    if not new_download:
        pass
    else:
        metadata_filename = "data/MRI_Duke_Image_Data/metadata.csv"
        annot_filename = "data/MRI_Duke_Image_Data/Annotation_Boxes.csv"
        df__metadata = pd.read_csv(metadata_filename)
        df__annot = pd.read_csv(annot_filename)
        root_path = os.getcwd()
        root_path = root_path.replace("//", "/")
        for index, row in df__metadata.iterrows():
            PID = row["Subject ID"]
            file_location = row["File Location"]
            file_location = file_location.replace("//","/").lstrip(".")
            file_location = root_path + "/data/MRI_Duke_Image_Data" + file_location
            if "segment" in file_location:
                os.replace(file_location, "data/MRIDID/{}/Segmentation/".format(PID))
                continue
            imgs = os.listdir(file_location)
            something = df__annot.loc[df__annot["Patient ID"] == PID] # Check if images are correctly allocated
            start = int(something["Start Slice"]) - 1
            end = int(something["End Slice"]) - 1
            sel_imgs = imgs[start:end]
            if not sel_imgs:
                continue
            # Move images to another location
            for img in sel_imgs:
                src = os.path.join(file_location, img)
                new_file_location1 = root_path + "/data/MRIDID/{}/".format(PID)
                new_file_location2 = root_path + "/data/MRIDID/{}/{}/".format(PID, row["Series Description"])
                try:
                    os.mkdir(new_file_location1)
                except FileExistsError:
                    pass
                try:
                    os.mkdir(new_file_location2)
                except FileExistsError:
                    pass
                try:
                    Path(src).rename(os.path.join(new_file_location2, img))
                except FileExistsError:
                    pass

def obtain_data(filename:str):
    """Extract the data from the .dcm files.

    ...

    Loads the data using the pydicom library to extract both metadata and more.
    """
    ds = pydicom.dcmread(filename)
    slices = ds.pixel_array
    sex = ds.PatientSex
    if sex == "F":
        sex = 1
    else:
        sex = 2
    age = ds.PatientAge
    age = age.replace("Y", "")
    age = int(age)
    weight = int(ds.PatientWeight)
    modality = ds.Modality
    if modality == 'MR':
        modality = 1
    elif modality == 'CT':
        modality = 2
    elif modality == 'PT':
        modality = 3
    else:
        modality = 4
    pregnancy_status = ds.PregnancyStatus
    PID = ds.PatientID
    if slices.ndim <= 2:
        pass
    elif slices.ndim >= 3:
        slices = slices[0]
    datapoint = {"Patient ID":PID, "Image":slices, "metadata": [sex, age, weight, modality, pregnancy_status]}
    return datapoint

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


if __name__ == "__main__":
    main()
