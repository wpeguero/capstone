from keras.layers import Conv2D, MaxPooling2D, Dense, Rescaling, Flatten, GRU, Embedding, Layer, RandomFlip
import matplotlib.pyplot as plt
from keras.models import Model
import tensorflow as tf
from PIL import Image
import pandas as pd
import pathlib
import pydicom
import json

def main():
    #image_file_path = input("Please write relative path to image data set:")
    #image_type = input("\nPlease insert the image type for the dataset (jpg, jpeg, png):")
    #model = create_model(image_file_path, image_type)
    #model.save('models/MRILove-v2')
    df__metadata = pd.read_csv('data/lung_cancer/manifest-1600709154662/metadata.csv')
    file_locations = df__metadata['File Location'].tolist()
    file_locations = list(map(lambda x: x.replace("\\", "/"), file_locations))
    file_locations = list(map(lambda x:x.replace("./", "data/lung_cancer/manifest-1600709154662/"), file_locations))
    modalities = df__metadata['Modality']
    df__diagnosis = pd.read_excel('data/lung_cancer/tcia-diagnosis-data-2012-04-20.xls')
    with open("data/lung_cancer/tcia-diagnosis-legend.json") as fp:
        diagnosis_legend = json.load(fp)
        fp.close()
    print(diagnosis_legend)


def create_model(filename:str, img_type:str):
    """Train and create the model based on the data set."""
    data_dir = pathlib.Path(str(filename))
    image_count = len(list(data_dir.glob(f"*/*.{str(img_type)}")))
    BATCH_SIZE = 32
    BUFFER_SIZE = image_count
    EPOCHS = 100
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
    class_names = train_ds.class_names
    dataset = train_ds.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=BUFFER_SIZE)
    model = DoctorLove(
        img_height=img_height,
        img_width=img_width,
        class_names=class_names,
        rgb=True
        )
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(dataset, validation_data=val_ds, epochs=EPOCHS)
    return model


class DoctorLove(Model):
    def __init__(self, img_height:float,img_width:float, class_names:list, rgb:bool=False):
        super().__init__(self)
        if rgb == True:
            self.rescale = Rescaling(1./255, input_shape=(img_height, img_width, 3))
            self.conv1 = Conv2D(16, 3, padding='same', activation='relu')
            self.maxpool1 = MaxPooling2D()
            self.conv2 = Conv2D(32, 3, padding='same', activation='relu')
            self.maxpool2 = MaxPooling2D()
            self.conv3 = Conv2D(64, 3, padding='same', activation='relu')
            self.maxpool3 = MaxPooling2D()
            self.flatten = Flatten()
            self.dense1 = Dense(128, activation='relu')
            self.classify = Dense(len(class_names))
        else:
            self.rescale = Rescaling(1./255, input_shape=(img_height, img_width, 1))
            self.conv1 = Conv2D(16, 1, padding='same', activation='relu')
            self.maxpool1 = MaxPooling2D()
            self.conv2 = Conv2D(32, 1, padding='same', activation='relu')
            self.maxpool2 = MaxPooling2D()
            self.conv3 = Conv2D(64, 1, padding='same', activation='relu')
            self.maxpool3 = MaxPooling2D()
            self.flatten = Flatten()
            self.dense1 = Dense(128, activation='relu')
            self.classify = Dense(len(class_names))
    
    def call(self, inputs):
        x = inputs
        x = self.rescale(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.classify(x)
        return x


class BahnaduaAttention(Model):
    def __init__(self, units):
        super().__init__(self)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.v = Dense(1)

    def call(self, features, hidden):
        #Features(CaptionImageEncoder) shape == (batch_size, 64, embedding_dim)
        pass


class CaptionImageEncoder(Model):
    def __init__(self, embedding_dim):
        super().__init__(self)
        self.fc = Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class CaptionImageDecoder(Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super().__init__(self)
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.units, return_sequences=True, return_state=True, return_initializer='glorot_uniform')
        self.fc1 = Dense(self.units)
        self.fc2 = Dense(vocab_size)


class Augment(Layer):
    def __init__(self, seed=42):
        super().__init__()
        # Both use the same seed, they'll make the same random changes.
        self.augment_inputs = RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = RandomFlip(mode="horizontal", seed=seed)
    
    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


if __name__ == "__main__":
    main()
