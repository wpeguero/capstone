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
import pathlib
import tensorflow as tf
import pydicom
from pydicom.errors import InvalidDicomError
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def main():
    """Test the new functions."""
    filename = "data/CMMD-set/medical_data_with_image_paths.csv"
    #_convert_dicom_to_png(filename)
    #df__img = _extract_key_images(data_dir="/data/CMMD-set", metadata_filename="data/CMMD-set/metadata.csv", new_download=True)
    #df = pd.read_excel('data/CMMD-set/CMMD_clinicaldata_revision.xlsx')
    #df_merge = df__img.merge(df, how='left', left_on='Subject ID', right_on='ID1')
    #df_merge.to_csv(filename, index=False)
    #directory = "data/CMMD-set/classifying_set"
    #tset, valset = load_data(directory)
    filename = "data/CMMD-set/CMMD/D1-0001/07-18-2010-NA-NA-79377/1.000000-NA-70244/1-1.dcm"
    print(extract_data(filename))


def _convert_dicom_to_png(filename:str):
    """Convert a list of dicom files into their png forms.
    
    ...
    """
    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        ds = pydicom.dcmread(row['paths'])
        path = pathlib.PurePath(row['paths'])
        dicom_name = path.name
        name = dicom_name.replace(".dcm", "")
        new_image = ds.pixel_array.astype(float)
        scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255
        scaled_image = np.uint8(scaled_image)
        final_image = Image.fromarray(scaled_image)
        final_image.save(f"data/CMMD-set/classifying_set/{row['classification']}/{row['Subject ID'] + '_' + name}.png")

def _extract_key_images(data_dir:str, metadata_filename:str, new_download = False):
    """Extract the key images based on the Annotation Boxes file.
    
    Grabs the images from the full directory and
    moves them to a separate directory for keeping
    only the key data."""
    if not new_download:
        return None
    else:
        df__metadata = pd.read_csv(metadata_filename)
        root_path = os.getcwd()
        root_path = root_path.replace("//", "/")
        img_paths_list = list()
        for _, row in df__metadata.iterrows():
            PID = row["Subject ID"]
            file_location = row["File Location"]
            file_location = file_location.replace("//","/").lstrip(".")
            file_location = root_path + data_dir + file_location
            imgs = os.listdir(file_location)
            for img in imgs:
                img_paths = {
                    'Subject ID': PID,
                    'paths': file_location + '/' + img
                }
                img_paths_list.append(img_paths)
        df_img_paths = pd.DataFrame(img_paths_list)
        return df_img_paths

def extract_data(filename:str):
    """Extract the data from the .dcm files.

    ...

    Reads each independent file using the pydicom
    library and extracts key information, such as
    the age, sex, ethnicity, weight of the patient,
    and the imaging modality used.

    Parameter(s)
    ------------

    filename`str`: path to the file (either relative or absolute).
        - The file must end in .dcm or else an error will be thrown.
    """
    try:
        ds = pydicom.dcmread(filename)
    except (InvalidDicomError) as e:
        print(f"ERROR: The file {filename} is not a DICOM file and therefore cannot be read.")
        exit()
    datapoint = dict()
    slices = ds.pixel_array
    try:
        sex = ds.PatientSex
        datapoint['sex'] = sex
    except AttributeError as e:
        pass
    try:
        age = ds.PatientAge
        datapoint['age'] = age
    except AttributeError as e:
        pass
    try:
        weight = ds.PatientWeight
        datapoint['weight'] = weight
    except AttributeError as e:
        pass
    try:
        modality = ds.Modality
        datapoint['modality'] = modality
    except AttributeError as e:
        pass
    PID = ds.PatientID
    if slices.ndim <= 2:
        pass
    elif slices.ndim >= 3:
        slices = slices[0]
    datapoint['Subject ID'] = PID
    datapoint['Image'] = slices
    return datapoint

def transform_data(datapoint:dict):
    """ Transform the data into an format that can be used for displaying and modeling.

    ...

    Grabs the extracted data and begins transforming
    the data into a format that can be used for
    display in a dashboard as well as for modeling
    purposes.
    """
    try:
        if datapoint['sex'] == 'F':
            datapoint['sex'] = 1
        elif datapoint['sex'] == 'M':
            datapoint['sex'] = 2
        else:
            sex = 0
    except (AttributeError, KeyError) as e:
        print('WARNING: Indicator "sex" does not exist.')
    
    try:
        if "Y" in datapoint['age']:
            datapoint['age'] = datapoint['age'].replace('Y', '')
        else:
            pass
        datapoint['age'] = int(datapoint['age'])
    except (AttributeError, KeyError) as e:
        print('WARNING: Indicator "age" does not exist.')
    
    try:
        datapoint['weight'] = int(datapoint['weight'])
    except (AttributeError, KeyError) as e:
        #print('WARNING: Indicator "weight" does not exist.')
        pass
    
    try:
        if datapoint['modality'] == 'MR':
            datapoint['modality'] = 1
        elif datapoint['modality'] == 'CT':
            datapoint['modality'] = 2
        elif datapoint['modality'] == 'PT':
            datapoint['modality'] = 3
        else:
            datapoint['modality'] = 0
    except (AttributeError, KeyError) as e:
        print('WARNING: Indicator "modality" does not exist.')
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
    data_dir = pathlib.Path(directory)
    image_count = len(list(data_dir.glob(f"*/*")))
    BATCH_SIZE = 32
    BUFFER_SIZE = image_count
    img_height = 180
    img_width = 180
    train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=(img_height, img_width),
            batch_size=BATCH_SIZE
            )
    dataset = train_ds.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    main()
