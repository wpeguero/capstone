"""This file will contain all of the acutalized models created from the abstract model class(es) made within the base.py file."""
from keras.layers import Conv2D, Dense, Rescaling, Flatten, MaxPooling2D, Dropout, RandomZoom, AveragePooling2D, Input, concatenate
from keras.models import Model
from keras.utils import plot_model
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

def main():
    inputs, output = tumor_classifier(448,448)
    model = Model(inputs=inputs, outputs=output)
    #model.build(input_shape=(800,800))
    #plot_model(model, show_shapes=True)
    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=[SparseCategoricalAccuracy()])
    model.save("tumorclassifier/my_model")


class BaseImageClassifier(Model):
    """Basic Image Classifier for model comparison improvement.

    ...

    A class containing a simple classifier for any
    sort of image. The models stemming from this
    class will function to only classify the image
    in one manner alone (malignant or non-malignant).
    This model will not contain any rescaling or 
    data augmentation to show how significant the
    accuracy between a model with rescaling and
    data augmentation is against a model without
    any of these.
    """

    def __init__(self, img_height:float, img_width:float):
        """Initialize the layers for the model."""
        super().__init__(self)
        self.conv = Conv2D(16, 3, padding='same', activation='relu')
        self.avgpool = AveragePooling2D()
        self.flatten = Flatten()
        self.dropout = Dropout(0.3)
        self.dense = Dense(128, activation='relu')
        self.tumor = Dense(2, activation='softmax')

    def call(self, inputs):
        """Call the layers into action for training."""
        x = inputs
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        tumor = self.tumor(x)
        return tumor

class ImageClassifier(Model):
    """Basic Image Classifier with rescaling and data augmentation.

    ...

    A class containing a simple classifier for any
    sort of image. The models stemming from this class
    will include rescaling and data augmentation
    for the sake and purpose of normalizing the data.
    """
    
    def __init__(self, img_height:float, img_width):
        """Initialize the Layers of the model."""
        super().__init__(self)
        self.rescale = Rescaling(1./255, input_shape = (img_height, img_width, 3))
        self.augment = RandomZoom(0.1)
        self.conv = Conv2D(16, 3, padding='same', activation='relu')
        self.avgpool = AveragePooling2D()
        self.flatten = Flatten()
        self.dropout = Dropout(0.3)
        self.dense = Dense(128, activation='relu')
        self.tumor = Dense(2, activation='softmax')

    def call(self, inputs):
        """Call the layers into action for training."""
        x = inputs
        x = self.rescale(x)
        x = self.augment(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        tumor = self.tumor(x)
        return tumor

def tumor_classifier(img_height:float, img_width:float):
    """Complete Tumor Classification Algorithm.

    ...

    A class containing a simple classifier for any
    sort of image. The models stemming from this class
    will include rescaling and data augmentation
    for the sake and purpose of normalizing the data.
    """
    img_input = Input(shape=(img_height,img_width,3), name="Image Input")
    cat_input = Input(shape=(5), name="Categorical Input")
    inputs = [img_input, cat_input]
    # Set up the images
    x = Rescaling(1./255, input_shape= (img_height, img_width,3))(img_input)
    x = RandomZoom(0.1)(x)
    x = Conv2D(16, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    #Set up the categorical data
    y = Dense(1)(cat_input)
    # Merge both layers
    together = concatenate([x,y])
    output = Dense(2, activation='softmax')(together)
    return inputs, output


class TumorClassifier(Model):
    """Complete Tumor Classification Algorithm.

    ...

    A class containing a simple classifier for any
    sort of image. The models stemming from this class
    will include rescaling and data augmentation
    for the sake and purpose of normalizing the data.
    """
    
    def __init__(self, img_height:float, img_width):
        """Initialize the layers of the model."""
        super().__init__(self)
        self.rescale = Rescaling(1./255, input_shape = (img_height, img_width, 3))
        self.augment = RandomZoom(0.1)
        self.conv1 = Conv2D(16, 3, padding='same', activation='relu')
        self.avgpool1 = AveragePooling2D()
        self.conv2 = Conv2D(32, 3, padding='same', activation='relu')
        self.avgpool2 = AveragePooling2D()
        self.conv3 = Conv2D(64, 3, padding='same', activation='relu')
        self.avgpool3 = AveragePooling2D()
        self.flatten = Flatten()
        self.dropout = Dropout(0.3)
        self.dense = Dense(128, activation='relu')
        self.tumor = Dense(2, activation='softmax')

    def call(self, inputs):
        """Call the layers into action for training."""
        x = inputs
        x = self.rescale(x)
        x = self.augment(x)
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = self.avgpool3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        tumor = self.tumor(x)
        return tumor


if __name__ == "__main__":
    main()