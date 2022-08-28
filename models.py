"""This file will contain all of the acutalized models created from the abstract model class(es) made within the base.py file."""
from keras.layers import Conv2D, Dense, Rescaling, Flatten, MaxPooling2D, Dropout
from keras.models import Model

class BaseImageClassifier(Model):
    """Basic Image Classifier for model comparison improvement.

    ...

    A class containing a simple classifier for any
    sort of image. The models stemming from this
    class will function to only classify the image
    in one manner alone (malignant or non-malignant).
    To include the tumor stage, a separate version of
    the same model will may have to be created. This
    model will not contain any rescaling or data
    augmentation to show how significant the
    accuracy between a model with rescaling and
    data augmentation is against a model without
    any of these.
    """

    def __init__(self, img_height:float, img_width:float):
        """Initialize the layers for the model."""
        super().__init__(self)
        self.conv = Conv2D(16, 3, padding='same', activation='relu')
        self.maxpool = MaxPooling2D()
        self.flatten = Flatten()
        self.dropout = Dropout(0.3)
        self.dense = Dense(128, activation='relu')
        self.tumor = Dense(2, activation='softmax')
        self.stage = Dense(7, activation='softmax')

    def call(self, inputs):
        """Call the layers into action for training."""
        x = inputs
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        tumor = self.tumor(x)
        cancer_stage = self.stage(x)
        return cancer_stage, tumor

class ImageClassifier(Model):
    """Basic Image Classifier with rescaling and data augmentation.

    ...

    A class containing a simple classifier for any
    sort of image. The models stemming from this class
    will include rescaling and data augmentation
    for the sake and purpose of normalizing the data.
    """
