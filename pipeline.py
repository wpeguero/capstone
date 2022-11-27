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
import os
import pathlib
from fractions import Fraction

import numpy as np
import pandas as pd
from pydicom import dcmread
from PIL import Image
from pydicom.errors import InvalidDicomError
from tensorflow.keras.layers import CategoryEncoding
from tensorflow.keras.models import load_model
from tensorflow.nn import softmax
from tensorflow import data
from tensorflow import keras

#Base data
modalities = {
    0: 'MR',
    1: 'CT',
    2: 'PT',
    3: 'MG'
}

sides = {
    0: 'L',
    1: 'R'
}

sex = {
    0: 'F',
    1: 'M'
}

class_names = {
    0: 'Benign',
    1: 'Malignant'
}

##The dataset had duplicates due to images without any data provided on the clinical analysis. Some images were taken without clinical data for the purpose of simply taking the image. Nothing was identified for these and therefore these should be removed from  the dataset before converting the .dcm files into .png files.
def _main():
    """Test the new functions."""
    filename = "data/CMMD-set/test.csv"
    fpredictions = './test_predictions8.csv'
    if os.path.exists(fpredictions):
        dfp = pd.read_csv(fpredictions)
    else:
        mname = "./models/tclass_VGG4"
        dft = load_testing_data(filename)
        dfp = predict(dft, mname)
        dfp.to_csv(fpredictions, index=False) 
    df = pd.read_csv(filename)
    df = df.dropna(subset=['classification'])
    dfp = dfp.merge(df, left_on=['Subject ID', 'side'], right_on=['ID1', 'LeftRight']) #TODO Resolve the duplication issue when merging.
    dfp.to_csv('merged_for_confusion.csv')
    ct01 = pd.crosstab(dfp['pred_class'], dfp['classification'])
    print(ct01)


def _convert_dicom_to_png(filename:str) -> None:
    """Convert a list of dicom files into their png forms.
    
    ...
    """
    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        ds = dcmread(row['paths'])
        path = pathlib.PurePath(row['paths'])
        dicom_name = path.name
        name = dicom_name.replace(".dcm", "")
        new_image = ds.pixel_array.astype(float)
        scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255
        scaled_image = np.uint8(scaled_image)
        final_image = Image.fromarray(scaled_image)
        final_image.save(f"data/CMMD-set/classifying_set/raw_png/{row['Subject ID'] + '_' + name + ds.ImageLaterality}.png")

def _extract_key_images(data_dir:str, metadata_filename:str, new_download = False) -> pd.DataFrame:
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
                ds = dcmread(file_location + '/' + img)
                img_paths = {
                    'ID1': PID,
                    'paths': file_location + '/' + img,
                    'LeftRight': ds.ImageLaterality
                }
                img_paths_list.append(img_paths)
        df_img_paths = pd.DataFrame(img_paths_list)
        return df_img_paths

def extract_data(file) -> dict:
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
            ds = dcmread(file)
        except (InvalidDicomError) as e:
            print(f"ERROR: The file {file} is not a DICOM file and therefore cannot be read.")
            exit()
    else:
        ds = file
    datapoint = dict()
    slices = ds.pixel_array
    targetData = ['PatientSex', 'PatientAge', 'PatientWeight', 'Modality', 'ImageLaterality', 'PatientID']
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
    datapoint['image'] = slices
    return datapoint

def transform_data(datapoint:dict) -> dict:
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
        img = datapoint['image']
        img_mod = rescale_image(img)
        datapoint['image'] = img_mod
    except (AttributeError, KeyError) as e:
        print('WARNING: Indicator "image" does not exist.')
    return datapoint

def _balance_data(df:pd.DataFrame) -> pd.DataFrame:
    """Balance data for model training.
    
    Splits the dataset into groups based on the categorical
    columns provided. This function is hidden due to its
    ability being very specific to the project and only
    required during the loading stages of new training data.

    Parameter(s)
    ------------
    df : Pandas DataFrame
        Contains all of the data necessary to load the
        training data set.
    
    Returns
    -------
    df_balanced : Pandas DataFrame
        Balanced data set ready for feature extraction.
    """
    ccat = df['classification'].unique() # 2 categories
    scat = df['LeftRight'].unique() # 2 categories
    acat = df['abnormality'].unique() # 3 categories
    df_group1 = df.loc[(df['classification'] == ccat[0]) & (df['LeftRight'] == scat[0]) & (df['abnormality'] == acat[0])]
    df_group1 = df_group1.sample(n=58, random_state=42)
    df_group2 = df.loc[(df['classification'] == ccat[0]) & (df['LeftRight'] == scat[0]) & (df['abnormality'] == acat[1])]
    df_group2 = df_group2.sample(n=86, random_state=42)
    df_group3 = df.loc[(df['classification'] == ccat[0]) & (df['LeftRight'] == scat[0]) & (df['abnormality'] == acat[2])]
    df_group3 = df_group3.sample(n=110, random_state=42) # Total in group 406
    df_group4 = df.loc[(df['classification'] == ccat[0]) & (df['LeftRight'] == scat[1]) & (df['abnormality'] == acat[0])]
    df_group4 = df_group4.sample(n=70, random_state=42)
    df_group5 = df.loc[(df['classification'] == ccat[0]) & (df['LeftRight'] == scat[1]) & (df['abnormality'] == acat[1])]
    df_group5 = df_group5.sample(n=74, random_state=42)
    df_group6 = df.loc[(df['classification'] == ccat[0]) & (df['LeftRight'] == scat[1]) & (df['abnormality'] == acat[2])]
    df_group6 = df_group6.sample(n=110, random_state=42) # Total in group 418
    df_group7 = df.loc[(df['classification'] == ccat[1]) & (df['LeftRight'] == scat[0]) & (df['abnormality'] == acat[0])]
    df_group7 = df_group7.sample(n=110, random_state=42) # Total in group 190
    df_group8 = df.loc[(df['classification'] == ccat[1]) & (df['LeftRight'] == scat[0]) & (df['abnormality'] == acat[1])]
    df_group8 = df_group8.sample(n=110, random_state=42) # Total in group 344
    df_group9 = df.loc[(df['classification'] == ccat[1]) & (df['LeftRight'] == scat[0]) & (df['abnormality'] == acat[2])]
    df_group9 = df_group9.sample(n=110, random_state=42) # Total in group 684
    df_group10 = df.loc[(df['classification'] == ccat[1]) & (df['LeftRight'] == scat[1]) & (df['abnormality'] == acat[0])]
    df_group10 = df_group10.sample(n=110, random_state=42) # Total in group 206
    df_group11 = df.loc[(df['classification'] == ccat[1]) & (df['LeftRight'] == scat[1]) & (df['abnormality'] == acat[1])]
    df_group11 = df_group11.sample(n=110, random_state=42) # Total in group 418
    df_group12 = df.loc[(df['classification'] == ccat[1]) & (df['LeftRight'] == scat[1]) & (df['abnormality'] == acat[2])]
    df_group12 = df_group12.sample(n=110, random_state=42) # Total in group 790
    df_balanced = pd.concat([df_group1, df_group2, df_group3, df_group4, df_group5, df_group6, df_group7, df_group8, df_group9, df_group10, df_group11, df_group12], ignore_index=True)
    return df_balanced

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
    ds_img = keras.utils.image_dataset_from_directory(data_dir, batch_size=None, labels=None)
    print("This is just the image dataset {}.".format(data.experimental.cardinality(ds_img)))
    ds_cat = data.Dataset.from_tensor_slices((X_cat))
    print("This is just the categorical dataset {}.".format(data.experimental.cardinality(ds_cat)))
    y = data.Dataset.from_tensor_slices(y)
    X = data.Dataset.zip((ds_img, ds_cat))
    return X, y

def load_training_data(filename:str, first_training:bool=True, validate:bool=False):
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
    if (first_training == True and validate == True):
        df = pd.read_csv(filename)
        #Balancing the training data set
        df_balanced = _balance_data(df)
        df_test = df.drop(df_balanced.index)
        df_balanced.to_csv("./data/CMMD-set/train_dataset.csv", index=False)
        df = df_balanced
        # Balancing the validation data set
        vdf_balanced = _balance_data(df_test)
        df_test = df.drop(vdf_balanced.index)
        vdf_balanced.to_csv('./data/CMMD-set/validation_dataset.csv', index=False)
        df_test.to_csv("./data/CMMD-set/test_dataset.csv", index=False)
        vdf = vdf_balanced
        data = {
            'image': list(),
            'cat': list(),
            'class': list()
        }
        vdata = {
            'image': list(),
            'cat': list(),
            'class': list()
        }
        # Encode Training Data
        df['classification'] = pd.Categorical(df['classification'])
        df['enclassification'] = df['classification'].cat.codes
        df['LeftRight'] = pd.Categorical(df['LeftRight'])
        df['enLeftRight'] = df['LeftRight'].cat.codes
        # Encode Validation Data
        vdf['classification'] = pd.Categorical(vdf['classification'])
        vdf['enclassification'] = vdf['classification'].cat.codes
        vdf['LeftRight'] = pd.Categorical(vdf['LeftRight'])
        vdf['enLeftRight'] = vdf['LeftRight'].cat.codes
        simcoder = CategoryEncoding(num_tokens = 2, output_mode="one_hot")
        for (i, row), (j, vrow) in zip(df.iterrows(), vdf.iterrows()):
            # collect images
            ds = dcmread(row['paths'])
            vds = dcmread(vrow['paths'])
            img = ds.pixel_array
            vimg = vds.pixel_array
            img_mod = rescale_image(img)
            vimg_mod = rescale_image(vimg)
            # Collect the data
            data['image'].append(img_mod)
            data['cat'].append([row['enLeftRight'], row['Age']])
            data['class'].append(simcoder(row['enclassification']))

            vdata['image'].append(vimg_mod)
            vdata['cat'].append([vrow['enLeftRight'], vrow['Age']])
            vdata['class'].append(simcoder(vrow['enclassification']))
        data['image'] = np.asarray(data['image'])
        data['cat'] = np.asarray(data['cat'])
        data['class'] = np.asarray(data['class'])
        vdata['image'] = np.asarray(vdata['image'])
        vdata['cat'] = np.asarray(vdata['cat'])
        vdata['class'] = np.asarray(vdata['class'])
        return data, vdata
    elif (first_training == True and validate == False):
        df = pd.read_csv(filename)
        #Balancing the data set
        df_balanced = _balance_data(df)
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
            ds = dcmread(row['paths'])
            img = ds.pixel_array
            img_mod = rescale_image(img)
            #print(img_mod.shape)
            data['image'].append(img_mod)
            data['cat'].append([row['enLeftRight'], row['Age']])
            data['class'].append(simcoder(row['enclassification']))
        data['image'] = np.asarray(data['image'])
        data['cat'] = np.asarray(data['cat'])
        data['class'] = np.asarray(data['class'])
        return data, None
    elif (first_training == False and validate == True):
        df = pd.read_csv(filename)
        vdf = df.sample(n=200, random_state=42)
        data = {
            'image': list(),
            'cat': list(),
            'class': list()
        }
        vdata = {
            'image': list(),
            'cat': list(),
            'class': list()
        }
        # Encode Training Data
        df['classification'] = pd.Categorical(df['classification'])
        df['enclassification'] = df['classification'].cat.codes
        df['LeftRight'] = pd.Categorical(df['LeftRight'])
        df['enLeftRight'] = df['LeftRight'].cat.codes
        # Encode Validation Data
        vdf['classification'] = pd.Categorical(vdf['classification'])
        vdf['enclassification'] = vdf['classification'].cat.codes
        vdf['LeftRight'] = pd.Categorical(vdf['LeftRight'])
        vdf['enLeftRight'] = vdf['LeftRight'].cat.codes
        simcoder = CategoryEncoding(num_tokens = 2, output_mode="one_hot")
        for (i, row), (j, vrow) in zip(df.iterrows(), vdf.iterrows()):
            # collect images
            ds = dcmread(row['paths'])
            vds = dcmread(vrow['paths'])
            img = ds.pixel_array
            vimg = vds.pixel_array
            img_mod = rescale_image(img)
            vimg_mod = rescale_image(vimg)
            # Collect the data
            data['image'].append(img_mod)
            data['cat'].append([row['enLeftRight'], row['Age']])
            data['class'].append(simcoder(row['enclassification']))

            vdata['image'].append(vimg_mod)
            vdata['cat'].append([vrow['enLeftRight'], vrow['Age']])
            vdata['class'].append(simcoder(vrow['enclassification']))
        data['image'] = np.asarray(data['image'])
        data['cat'] = np.asarray(data['cat'])
        data['class'] = np.asarray(data['class'])
        vdata['image'] = np.asarray(vdata['image'])
        vdata['cat'] = np.asarray(vdata['cat'])
        vdata['class'] = np.asarray(vdata['class'])
        return data, vdata
    else:
        df = pd.read_csv(filename)
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
            ds = dcmread(row['paths'])
            img = ds.pixel_array
            img_mod = rescale_image(img)
            #print(img_mod.shape)
            data['image'].append(img_mod)
            data['cat'].append([row['enLeftRight'], row['Age']])
            data['class'].append(simcoder(row['enclassification']))
        data['image'] = np.asarray(data['image'])
        data['cat'] = np.asarray(data['cat'])
        data['class'] = np.asarray(data['class'])
        return data, None

def  load_testing_data(filename:str) -> pd.DataFrame: #Shrink the images from their full size
    """Load the data used  for testing.
    
    Loads a dataset to be fed into the model for making
    predictions. The output of the testing data will be
    comprised of a dictionary that can be fed directly into
    the model.

    Parameter(s)
    ------------
    filename : str
        path to file containing the file paths to test data.
    """
    df = pd.read_csv(filename)
    df = df.dropna(subset=['classification'])
    print("iterating through {} rows...".format(len(df)))
    dfp_list = list()
    for _, row in df.iterrows():
        datapoint = extract_data(row['paths'])
        datapoint = transform_data(datapoint)
        dfp_list.append(datapoint)
    tdata = pd.DataFrame(dfp_list)
    return tdata

def predict(data:pd.DataFrame, model_name:str) -> pd.DataFrame:
    """Make predictions based on dataset.

    Extracts the image data and required categories
    for loading into the model.
    """
    model = load_model(model_name)
    fdata = {'image': np.asarray(data['image'].to_list()), 'cat': np.asarray(data[['age', 'side']])}
    predictions = model.predict(fdata, batch_size=5)
    data['sex'] = data['sex'].map(sex)
    data['modality'] = data['modality'].map(modalities)
    data['side'] = data['side'].map(sides)
    if len(predictions) < 2 and len(predictions) > 0:
        predictions = predictions[0]
        data['score'] = [softmax(predictions).numpy()]
        data['pred_class'] = class_names[np.argmax(data['score'])]
    elif len(predictions) >= 2:
        predictions = predictions
        pred_data = list()
        for pred in predictions:
            score = softmax(pred)
            pclass = class_names[np.argmax(score)]
            pred_data.append({'score':score.numpy(), 'pred_class':pclass})
        _df = pd.DataFrame(pred_data)
        data = data.join(_df)
    data = data.drop(columns=['image'])
    return data

def rescale_image(img:np.ndarray) -> np.ndarray:
    """Rescale the image to a more manageable size.
    
    Changes the size of the image based on the length and
    width of the image itself. This is to reduce the amount
    of computations required to make predictions based on
    the image.
    
    Parameter(s)
    ------------
    img : Numpy Array
        array containing the raw values of images.
    """
    size = img.shape
    frac = Fraction(size[1], size[0])
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
    return img_mod

def calculate_confusion_matrix(fin_predictions:pd.DataFrame):
    """Calculate the confusion matrix using pandas.
    
    Calculates the confusion matrix using a csv file that
    contains both the predictions and actual labels. This
    function then creates a crosstab of the data to develop
    the confusion matrix.
    
    Parameter(s)
    ------------
    fin_predictions : Pandas DataFrame
        DataFrame containing the prediction and actual
        labels.
    
    Returns
    -------
    ct : Pandas DataFrame
        Cross tab containing the confusion matrix of the
        predictions compared to the actual labels.
    
    metrics : Dictionary
        Contains the basic metrics obtained from the
        confusion matrix. The metrics are the following:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
    """
    ct = pd.crosstab(fin_predictions['pred_class'], fin_predictions['classification'])
    # Set the initial values
    tp = ct[1][1]
    tn = ct[0][0]
    fp = ct[0][1]
    fn = ct[1][0]
    # Calculate the metrics
    metrics = dict()
    metrics['Accuracy'] = (tp + tn) / (tp + tn + fp + fn) # Ability of model to get the correct predictions
    metrics['Precision'] = tp / (tp + fp) # Ability of model to label actual positives as positives (think retrospectively)
    metrics['Recall'] = tp / (tp + fn) # Ability of model to correctly identify positives
    metrics['F1 Score'] = (2 * metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall'])
    return ct, metrics


if __name__ == "__main__":
    _main()
