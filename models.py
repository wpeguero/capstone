"""This file will contain all of the actualized models created from the abstract model class(es) made within the base.py file."""
from tensorflow.keras.layers import Conv2D, Dense, Rescaling, Flatten, MaxPooling2D, Dropout, RandomZoom, Permute, Reshape, Input, Concatenate
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, split_dataset
from tensorflow.saved_model import save
import tensorflow as tf
from pipeline import load_data, load_data2

BATCH_SIZE = 7

def main():
    inputs, output = tumor_classifier(1147, 957, BATCH_SIZE)
    model = Model(inputs=inputs, outputs=output)
    #model.build(input_shape=(800,800))
    plot_model(model, show_shapes=True)
    
    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
    filename = "data/CMMD-set/clinical_data_with_unique_paths.csv"
    data = load_data2(filename)
    y = data['class']
    data.pop('class')
    #print("This is the number of samples within the dataset {}.".format(len(y)))
    #X, y = load_data(filename, batch_size=BATCH_SIZE)
    #Xy = tf.data.Dataset.zip((X,y)).cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    #Xylength = tf.data.experimental.cardinality(Xy)
    #Xlength = tf.data.experimental.cardinality(X)
    #ylength = tf.data.experimental.cardinality(y)
    #print(Xylength.numpy())
    #print(Xlength.numpy())
    #print(ylength.numpy())
    dataset = tf.data.Dataset.from_tensor_slices((data, y))
    tf.data.Dataset.save(dataset, 'data/CMMD-set/saved_data2')
    exit()
    dataset = tf.data.Dataset.load('data/CMMD-set/saved_data')
    model.fit(dataset, epochs=10, batch_size=BATCH_SIZE)
    save(model,'tclass_V1')


def base_image_classifier(img_height:float, img_width:float):
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

    Parameters
    -----------
    img_height : float
        The height, in pixels, of the input images.
        This can be the maximum height of all images
        within the dataset to fit a varied amount
        that is equal or less than the declared height.
    
    img_width : float
        The width, in pixels, of the input images.
        This can also be the maximum width of all
        images within the dataset to fit a varied
        amount that is equal or smaller in width
        to the declared dimension.
    
    batch_size : int
        One of the factors of the total sample size.
        This is done to better train the model without
        allowing the model to memorize the data.
    
    Returns
    -------
    inputs : {img_input, cat_input}
        Input layers set to receive both image and
        categorical data. The image input contains
        images in the form of a 2D numpy array. The
        categorical input is a 1D array containing
        patient information. This is mainly comprised
        of categorical data, but some nominal data.
    
    output : Dense Layer
        The last layer of the model developed. As
        the model is fed through as the input of
        the next layer, the last layer is required
        to create the model using TensorFlow's Model
        class.
    """
    img_input = Input(shape=(img_height,img_width,3), name="Image Input")
    cat_input = Input(shape=None, name="Categorical Input")
    inputs = [img_input, cat_input]
    # Set up the images
    x = Conv2D(16, 3, padding='same', activation='relu')(img_input)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    #x = Dense(128, activation='relu')(x)
    #Set up the categorical data
    y = Dense(2, activation='relu')(cat_input)
    y = Dense(1, activation='relu')(y)
    # Merge both layers
    together = Concatenate(axis=1)([x,y])
    output = Dense(2, activation='softmax', name="output")(together)
    return inputs, output

def base_image_classifier2(img_height:float, img_width:float):
    """Basic Image Classifier with rescaling and data augmentation.

    ...

    A class containing a simple classifier for any
    sort of image. The models stemming from this class
    will include rescaling and data augmentation
    for the sake and purpose of normalizing the data.
    
    Parameters
    -----------
    img_height : float
        The height, in pixels, of the input images.
        This can be the maximum height of all images
        within the dataset to fit a varied amount
        that is equal or less than the declared height.
    
    img_width : float
        The width, in pixels, of the input images.
        This can also be the maximum width of all
        images within the dataset to fit a varied
        amount that is equal or smaller in width
        to the declared dimension.
    
    batch_size : int
        One of the factors of the total sample size.
        This is done to better train the model without
        allowing the model to memorize the data.
    
    Returns
    -------
    inputs : {img_input, cat_input}
        Input layers set to receive both image and
        categorical data. The image input contains
        images in the form of a 2D numpy array. The
        categorical input is a 1D array containing
        patient information. This is mainly comprised
        of categorical data, but some nominal data.
    
    output : Dense Layer
        The last layer of the model developed. As
        the model is fed through as the input of
        the next layer, the last layer is required
        to create the model using TensorFlow's Model
        class.
    """
    img_input = Input(shape=(img_height,img_width,3), name="Image Input")
    cat_input = Input(shape=(5), name="Categorical Input")
    inputs = [img_input, cat_input]
    # Set up the images
    x = Rescaling(1./255, input_shape= (img_height, img_width,3))(img_input)
    x = RandomZoom(0.1)(x)
    x = Conv2D(16, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    #x = Dense(128, activation='relu')(x)
    #Set up the categorical data
    y = Dense(2, activation='relu')(cat_input)
    y = Dense(1, activation='relu')(y)
    # Merge both layers
    together = Concatenate(axis=1)([x,y])
    output = Dense(2, activation='softmax')(together)
    return inputs, output

def tumor_classifier(img_height:float, img_width:float, batch_size:int):
    """Complete Tumor Classification Algorithm.

    ...

    A class containing a simple classifier for any
    sort of image. The models stemming from this class
    will include rescaling and data augmentation
    for the sake and purpose of normalizing the data.

    Parameters
    -----------
    img_height : float
        The height, in pixels, of the input images.
        This can be the maximum height of all images
        within the dataset to fit a varied amount
        that is equal or less than the declared height.
    
    img_width : float
        The width, in pixels, of the input images.
        This can also be the maximum width of all
        images within the dataset to fit a varied
        amount that is equal or smaller in width
        to the declared dimension.
    
    batch_size : int
        One of the factors of the total sample size.
        This is done to better train the model without
        allowing the model to memorize the data.
    
    Returns
    -------
    inputs : {img_input, cat_input}
        Input layers set to receive both image and
        categorical data. The image input contains
        images in the form of a 2D numpy array. The
        categorical input is a 1D array containing
        patient information. This is mainly comprised
        of categorical data, but some nominal data.
    
    output : Dense Layer
        The last layer of the model developed. As
        the model is fed through as the input of
        the next layer, the last layer is required
        to create the model using TensorFlow's Model
        class.
    """
    img_input = Input(shape=(img_height, img_width, 1), batch_size=batch_size, name='image')
    cat_input = Input(shape=(2), batch_size=batch_size, name='cat')
    inputs = [img_input, cat_input]
    # Set up the images
    x = Rescaling(1./255, input_shape=(img_height, img_width,1))(img_input)
    #x = RandomZoom(0.1)(x)
    x = Conv2D(16, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    #x = Dense(512, activation='relu')(x)
    #x = Dense(256, activation='relu')(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dense(64, activation='relu')(x)
    #x = Dense(32, activation='relu')(x)
    #x = Dense(16, activation='relu')(x)
    #x = Dense(8, activation='relu')(x)
    #Set up the categorical data
    y = Dense(2, activation='relu')(cat_input)
    y = Dense(1, activation='relu')(y)
    # Merge both layers

    together = Concatenate(axis=1)([x,y])
    output = Dense(2, activation='sigmoid', name='class')(together)
    return inputs, output


if __name__ == "__main__":
    main()