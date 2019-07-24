import os
import sys
import keras
import keras_vggface
from pathlib import Path, PureWindowsPath
from keras.preprocessing import image
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model, model_from_json
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras.backend as K
import numpy as np
import pandas as pd
import mtcnn
from PIL import Image

def prepare_frame(files):
    """ Prepare the data for dataframe.
    
    Create 2 lists one containing the file locations and the other
    containing the relative ratings.
    """
    imagefiles = []
    ratingslist = []
    for file in files:
        data = pd.read_csv(file, header = None)

        dirs = data.iloc[:,0].tolist()
        files = data.iloc[:,0].tolist()
        ratings = data.iloc[:,1].tolist()

        # find . in filename to split data
        i = 0
        for letter in dirs[0]:
            i += 1
            if letter == '.':
                break

        # check if image already in list and if not append to with ratings as list
        # so that we can calculate the average
        for j in range(len(dirs)):
            dirs[j] = PureWindowsPath(dirs[j][:i-1])
            dirs[j] = Path(dirs[j])
            files[j] = files[j][i:]
            imagefile = os.path.join(dirs[j], files[j])
            image_already_in_list = False
            for location in range(len(imagefiles)):
                if imagefiles[location] == imagefile:
                    image_already_in_list = True
                    break
            if not image_already_in_list:
                imagefiles.append(imagefile)
                ratingslist.append([ratings[j]])
            else:
                ratingslist[location].append(ratings[j])

    ratingslist = get_avg_ratings(ratingslist)
    return imagefiles, ratingslist

def get_avg_ratings(ratingslist):
    """ Calcuate the average rating of each image. """
    for j in range(len(ratingslist)):
        i = 0
        avg = 0
        for rating in ratingslist[j]:
            avg += rating
            i += 1
        ratingslist[j] = avg/i/10

    return ratingslist

def create_dataframe(files):
    """ Create the dataframe containing the info about the images and ratings. """
    imagefiles, ratingslist = prepare_frame(files)
    ratingframe = pd.DataFrame({'files': imagefiles, 'ratings': ratingslist})
    return ratingframe

def prepare_image(file):
    """ Prepare image to train model with.
    
    The image will be resized to size 224x224 and preprocessed
    using keras and numpy functions.
    """
    try:
        prepared = image.load_img(file, target_size=(224, 224))
        prepared = image.img_to_array(prepared)
        prepared = np.expand_dims(prepared, axis=0)
        prepared = preprocess_input(prepared)
    except Exception:
        raise ValueError
    return prepared[0]

def create_model(neurons=224, activation='sigmoid'):
    """ Create the transfer learning model. """
    base_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x=Dense(neurons,activation='relu')(x)
    #final layer
    preds=Dense(1,activation=activation)(x)

    model=Model(inputs=base_model.input,outputs=preds)
    return model

def train_model(model, data, epochs=20, batch_size=32, checkpoint=None):
    """ Return trained model and train history.

    Trains a model for given data and optionally epochs. 
    """
    # Load training images and tagets into numpy array obejct.
    images = []
    targets = []
    for d in range(len(data.iloc[:,0])):
        images.append(prepare_image(data.iloc[d,0]))
        targets.append([data.iloc[d,1]])
    images = np.array(images)
    targets = np.array(targets)

    model.compile(optimizer='Adam', loss='mse')
    if checkpoint:
        name = checkpoint
        filepath = "./weights/"+name+"best.hdf5"
        checkpoint = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1, patience=250)
        model.fit(images, targets, validation_split=0.25, callbacks=[checkpoint], epochs=epochs, batch_size=batch_size, shuffle=True)
    else:
        model.fit(images, targets, validation_split=0.25, epochs=epochs, batch_size=batch_size, shuffle=True)
    return model

def load_model(file):
    """ Load a model from json file. """
    json_file = open("file", 'r')
    model = json_file.read()
    json_file.close()
    model = model_from_json(model)

    return model
    
def save_model(model, file):
    """ Save a model to json file. """
    with open(file, 'w') as f:
        f.write(model.to_json())

def test_on_batch(model, files):
    """ Test model on files in dir or pass through list. 
    
    If you want to change color_space more changes to the code have to be made.
    """
    batch = []
    if isinstance(files, str):
        files = get_files_in_dir(files)
    for file in files:
        batch.append(prepare_image(file))
    batch = np.array(batch)
    preds = model.predict_on_batch(batch)

    return preds

def main():
    files = sys.argv[1:]
    train = True
    name = "first_try"

    model = create_model()
    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[19:]:
        layer.trainable = True
    
    data = create_dataframe(files)

    if train:
        model = train_model(model, data, checkpoint=name, epochs=5, batch_size=20)
        model.save_weights('weights/' + name + '.h5')
    else:
        model.load_weights("first_try.h5")

    predictions = test_on_batch(model, data.iloc[:,0])
    
    for i in range(len(predictions)):
        print(data.iloc[i,0], predictions[i]*10)

if __name__ == "__main__":
    main()
