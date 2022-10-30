"""Pipeline Module
------------------

Algorithms used to process data before modeling.

...

A set of algorithms used to feed in and process
data before used within the model. This will contain
the data extraction from its rawest form and output
the final form of the data set. The main source of
data will be image related from the Cancer Imaging
Archive.
"""
from inspect import Attribute
import pandas as pd
import os
import pathlib
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from tensorflow.nn import softmax
from tensorflow.keras.layers import CategoryEncoding
from numpy import argmax
import pydicom
from pydicom.errors import InvalidDicomError
import numpy as np
from PIL import Image
from fractions import Fraction
import shutil

##The dataset had duplicates due to images without any data provided on the clinical analysis. Some images were taken without clinical data for the purpose of simply taking the image. Nothing was identified for these and therefore these should be removed from  the dataset before converting the .dcm files into .png files.
def _main():
    """Test the new functions."""
    #model = tf.keras.models.load_model('tclass_V1/')
    #filename = 'data/CMMD-set/CMMD/D1-0001/07-18-2010-NA-NA-79377/1.000000-NA-70244/1-1.dcm'
    #filename2 = 'data/CMMD-set/CMMD/D1-0001/07-18-2010-NA-NA-79377/1.000000-NA-70244/1-2.dcm'
    #datapoint = extract_data(filename)
    #datapoint2 = extract_data(filename2)
    #datapoint = transform_data(datapoint)
    #datapoint2 = transform_data(datapoint2)
    #cat = np.array([datapoint['side'], datapoint['age']])
    #cat2 = np.array([datapoint2['side'], datapoint2['age']])
    #datapoint['Image'] = np.array([datapoint['Image']])
    #datapoint2['Image'] = np.array([datapoint2['Image']])
    #datapoint['Image'] = np.moveaxis(datapoint['Image'],0, -1)
    #datapoint2['Image'] = np.moveaxis(datapoint2['Image'],0, -1)
#
    #predictions = model({'image':np.array([datapoint['Image'], datapoint2['Image']]), 'cat':np.array([cat, cat2])})
    #print(predictions)
    #print(predictions[0].numpy())
    #score = softmax(predictions[0])
    #class_names = {0:'Benign', 1:'Malignant'}
    #print(class_names[argmax(score)], 100 * max(score.numpy()))
    #print(len(predictions))
    filename = "data/CMMD-set/test_dataset.csv"
    df = pd.read_csv(filename)
    df = df.sample(frac=1, random_state=42).reset_index()
    basedir = "./data/CMMD-set/sample_data/"
    #for i, row in df.iterrows():
    #    if i == 100:
    #        break
    #    else:
    #        pass
    #    print(i)
    #    fname = pathlib.Path(row['paths']).name
    #    shutil.copy2(row['paths'], '{}{}_{}'.format(basedir, row['ID1'], fname))


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
        final_image.save(f"data/CMMD-set/classifying_set/raw_png/{row['Subject ID'] + '_' + name + ds.ImageLaterality}.png")

def _extract_key_images(data_dir:str, metadata_filename:str, new_download = False):
    """Extract the key images based on the Annotation Boxes file.
    
    ...

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
                ds = pydicom.dcmread(file_location + '/' + img)
                img_paths = {
                    'ID1': PID,
                    'paths': file_location + '/' + img,
                    'LeftRight': ds.ImageLaterality
                }
                img_paths_list.append(img_paths)
        df_img_paths = pd.DataFrame(img_paths_list)
        return df_img_paths

def extract_data(file):
    """Extract the data from the .dcm files.

    ...

    Reads each independent file using the pydicom
    library and extracts key information, such as
    the age, sex, ethnicity, weight of the patient,
    and the imaging modality used.

    Parameters
    ---------
    file : Unknown 
        Either the path to the file or the file itself.
        In the case that the .dcm file is already
        loaded, the algorithm will proceed to extract
        the data. Otherwise, the algorithm will load
        the .dcm file and extract the necessary data.
    
    Returns
    -------
    datapoint : dictionary
        Dictionary comprised of the image data
        (numpy array), and the metadata associated
        with the DICOM file as its own separate
        `key:value` pair. This only pertains to the
        patient data and NOT the metadata describing
        how the image was taken.
    
    Raises
    ------
    InvalidDicomError
        The file selected for reading is not a DICOM
        or does not end in .dcm. Set in place to
        stop the algorithm in the case that any other
        filetype is introduced. Causes an error to be
        printed and the program to exit.
    
    AttributeError
        Occurs in the case that the DICOM file does
        not contain some of the metadata used for
        classifying the patient. In the case that
        the metadata does not exist, then the model
        continues on with the classification and some
        plots may be missing from the second page.
    """
    if type(file) == str:
        try:
            ds = pydicom.dcmread(file)
        except (InvalidDicomError) as e:
            print(f"ERROR: The file {file} is not a DICOM file and therefore cannot be read.")
            exit()
    else:
        ds = file
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
    try:
        side = ds.ImageLaterality
        datapoint['side'] = side
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

    Parameters
    ----------
    datapoint : dictionary
        Contains the image and related metadata in
        `key:value` pair format.
    
    Returns
    -------
    datapoint :dictionary
        same dictionary with the categorical data
        transformed into numerical (from text).
    
    Raises
    ------
    AttributeError
        Indicator of the `key` does not exists.
    
    KeyError
        Indicator of the `key` does not exists.
    """
    try:
        if datapoint['sex'] == 'F':
            datapoint['sex'] = 0
        elif datapoint['sex'] == 'M':
            datapoint['sex'] = 1
        else:
            sex = 2
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
        if datapoint['side'] == 'L':
            datapoint['side'] = 0
        elif datapoint['side'] == 'R':
            datapoint['side'] = 1
        else:
            datapoint['side'] = 2
    except (AttributeError, KeyError) as e:
        print('WARNING: Indicator "laterality" does not exist.')

    try:
        datapoint['weight'] = int(datapoint['weight'])
    except (AttributeError, KeyError) as e:
        #print('WARNING: Indicator "weight" does not exist.')
        pass
    
    try:
        if datapoint['modality'] == 'MR':
            datapoint['modality'] = 0
        elif datapoint['modality'] == 'CT':
            datapoint['modality'] = 1
        elif datapoint['modality'] == 'PT':
            datapoint['modality'] = 2
        else:
            datapoint['modality'] = 3
    except (AttributeError, KeyError) as e:
        print('WARNING: Indicator "modality" does not exist.')
    
    try:
        img = datapoint['Image']
        size = img.shape
        frac = Fraction(size[1], size[0]) #Width / Height
        width = frac.numerator
        height = frac.denominator
        img = img.astype(float)
        scaled_image = (np.maximum(img, 0) / img.max()) * 255
        scaled_image = np.uint8(scaled_image)
        final_image = Image.fromarray(scaled_image)
        final_image = final_image.resize(size=(width, height))
        img_mod = np.array(final_image)
        img_mod = np.asarray([img_mod])
        img_mod = np.moveaxis(img_mod, 0, -1)
        datapoint['Image'] = img_mod
    except (AttributeError, KeyError) as e:
        print('WARNING: Indicator "image" does not exist.')
    return datapoint

def load_data(filename:str, batch_size:int):
    """Load the data using tensorflow data set library.
    
    ...

    Uses the os library and the TensorFlow Data
    api to load, batch, and process the data for
    training.

    Parameter
    ---------
    filename : str
        Leads to a file containing the paths to
        all of the DICOM files as well as metadata.
    
    batch_size : int
        Factor of the length of the data set.
    
    Returns
    -------
    X : TensorFlow Dataset
        Zipped dataset containing both image data
        and categorical data together.
    
    y : TensorFlow Dataset
        Data set containing the classifications of
        the data.
    """
    df = pd.read_csv(filename)
    df['classification'] = pd.Categorical(df['classification'])
    df['classification'] = df['classification'].cat.codes
    y = df['classification']
    X_cat = df[['Age', 'LeftRight']]
    X_cat['LeftRight'] = pd.Categorical(X_cat['LeftRight'])
    X_cat['LeftRight'] = X_cat['LeftRight'].cat.codes
    data_dir = pathlib.Path('data/CMMD-set/classifying_set/raw_png/')
    ds_img = tf.keras.utils.image_dataset_from_directory(data_dir, batch_size=None, labels=None)
    print("This is just the image dataset {}.".format(tf.data.experimental.cardinality(ds_img)))
    ds_cat = tf.data.Dataset.from_tensor_slices((X_cat))
    print("This is just the categorical dataset {}.".format(tf.data.experimental.cardinality(ds_cat)))
    y = tf.data.Dataset.from_tensor_slices(y)
    X = tf.data.Dataset.zip((ds_img, ds_cat))
    return X, y

def load_data2(filename:str, batch_size:int=1):
    """Load the DICOM data as a dictionary.
    ...
    
    Creates a dictionary containing three different
    numpy arrays. The first array is comprised of
    multiple DICOM images, the second contains the
    categorical data as a vector, and the third contains
    the classification in numerical form.
    
    Parameters
    ----------
    filename : str
        path to a file which contains the metadata,
        classification, and path to the DICOM file.
        Will also contain some sort of ID to better
        identify the samples.
    
    batch_size : int
        Factor of the dataset size. Currently set
        to one as the standard for testing purposes.
    
    Returns
    -------
    data : dictionary
        Dictionary containing the encoded values
        for the metadata and the transformed image
        for input to the model.
    """
    df = pd.read_csv(filename)
    #Balancing the data set
    df_group1 = df.loc[(df['classification'] == 'Benign') & (df['LeftRight'] == 'L')]
    df_group1 = df_group1.sample(n=250, random_state=42)
    df_group2 = df.loc[(df['classification'] == 'Benign') & (df['LeftRight'] == 'R')]
    df_group2 = df_group2.sample(n=250, random_state=42)
    df_group3 = df.loc[(df['classification'] == 'Malignant') & (df['LeftRight'] == 'L')]
    df_group3 = df_group3.sample(n=250, random_state=42)
    df_group4 = df.loc[(df['classification'] == 'Malignant') & (df['LeftRight'] == 'R')]
    df_group4 = df_group4.sample(n=250, random_state=42)
    df_balanced = pd.concat([df_group1, df_group2, df_group3, df_group4], ignore_index=True)
    df_test = df.drop(df_balanced.index)
    df_balanced.to_csv("./data/CMMD-set/train_dataset.csv")
    df_test.to_csv("./data/CMMD-set/test_dataset.csv")
    df = df_balanced
    data = {
        'image': list(),
        'cat': list(),
        'class': list()
    }
    df['classification'] = pd.Categorical(df['classification'])
    df['enclassification'] = df['classification'].cat.codes
    df['LeftRight'] = pd.Categorical(df['LeftRight'])
    df['enLeftRight'] = df['LeftRight'].cat.codes
    simcoder = CategoryEncoding(num_tokens = 2, output_mode="one_hot")
    for i, row in df.iterrows():
        if i == 1000:
            break
        else:
            pass
        ds = pydicom.dcmread(row['paths'])
        img = ds.pixel_array
        size = img.shape
        frac = Fraction(size[1], size[0]) #Width / Height
        width = frac.numerator
        height = frac.denominator
        img = img.astype(float)
        scaled_image = (np.maximum(img, 0) / img.max()) * 255
        scaled_image = np.uint8(scaled_image)
        final_image = Image.fromarray(scaled_image)
        final_image = final_image.resize(size=(width, height))
        img_mod = np.asarray(final_image)
        img_mod = np.asarray([img_mod])
        img_mod = np.moveaxis(img_mod, 0, -1)
        #print(img_mod.shape)
        data['image'].append(img_mod)
        data['cat'].append([row['enLeftRight'], row['Age']])
        data['class'].append(simcoder(row['enclassification']))
    data['image'] = np.asarray(data['image'])
    data['cat'] = np.asarray(data['cat'])
    data['class'] = np.asarray(data['class'])
    return data


if __name__ == "__main__":
    _main()
