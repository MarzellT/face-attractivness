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
from glob import glob
import numpy as np
import pandas as pd
import mtcnn
from PIL import Image

def prepare_frame(files, use_actual_path=True, use_full_folder=None):
    """ Prepare the data for dataframe.
    
    Create 2 lists one containing the file locations and the other
    containing the relative ratings.
    If use_actual_path = True use the given file path instead of the
    original.
    if use_full_folder = True use all images in the data folder of the
    person. Will set use_actual_path to True.
    """
    imagefiles = []
    ratingsdict = {}
    fncnt = []
    if use_full_folder != None:
        use_actual_path = False
    for file in files:
        data = pd.read_csv(file, header = None)

        dirs = data.iloc[:,0].tolist()
        filenames = data.iloc[:,0].tolist()
        ratings = data.iloc[:,1].tolist()

        # find . in filename to split data between dirs and filenames
        i = 0
        for letter in dirs[0]:
            i += 1
            if letter == '.':
                break

        for j in range(len(dirs)):
            # setup path this
            # i figured this weird way is needed for this to work in windows (not tested though)
            dirs[j] = PureWindowsPath(dirs[j][:i-1])
            dirs[j] = Path(dirs[j])
            dirname = os.path.basename(dirs[j])
            basefoldername = os.path.dirname(os.path.dirname(dirs[j]))
            basefoldername = os.path.join(basefoldername, 'test')

            dirrating = ratings[j]
            curdir = os.path.join(basefoldername, dirname)

            fncnt.append(curdir)
            try:
                # here we take every image of the folder and assign the rating of
                # the one rated picture to all based on assumption described in the
                # README file
                for filename in os.listdir(curdir):
                    imagefile = os.path.join(curdir, filename)
                    ratingsdict = add_to_dict(imagefile, dirrating, ratingsdict)
            except Exception:
                print('error:', os.path.join(basefoldername, imagefolder))
        # check if image already in list and if not append to with ratings as list
        # so that we can calculate the average
        #for j in range(len(dirs)):
        #    if use_actual_path:
        #        dirs[j] = PureWindowsPath(dirs[j][:i-1] + filenames[j][i:])
        #        dirs[j] = Path(dirs[j])
        #        imagefile = dirs[j]
        #        ratingsdict = add_to_dict(imagefile, ratings[j], ratingsdict)
        #    else:
        #        dirs[j] = PureWindowsPath(dirs[j][:i-1])
        #        dirs[j] = Path(dirs[j])
        #        if use_full_folder:
        #            filename = os.path.basename(dirs[j])
        #            basefoldername = os.path.dirname(os.path.dirname(dirs[j]))
        #            basefoldername = os.path.join(basefoldername, 'test')
        #            for imagefolder in os.listdir(basefoldername):
        #                try:
        #                    for filename in os.listdir(os.path.join(basefoldername, imagefolder)):
        #                        fullfilename = os.path.join(imagefolder, filename)
        #                        imagefile = os.path.join(basefoldername, fullfilename)
        #                        ratingsdict = add_to_dict(imagefile, ratings[j], ratingsdict)
        #                except Exception:
        #                    print('error:', os.path.join(basefoldername, imagefolder))
        #            continue
        #        filenames[j] = filenames[j][i:]
        #        imagefile = os.path.join(dirs[j], filenames[j])
        #        print("wird ausgeführt")
        #        ratingsdict = add_to_dict(imagefile, ratings[j], ratingsdict)

    ratingsdict = get_avg_ratings(ratingsdict)
    print(len(ratingsdict))
    print(len(set(fncnt)))
    return ratingsdict


def add_to_dict(imagefile, rating, ratingsdict):
    """ Add the rating of the imagefile to the dictionary.

    If imagefile is not yet in the dictionary it gets added.
    Also add the first rating.
    If it is then just add the rating to the files ratingsdict.
    At the end return the updated dictionary.
    """
    try:
        ratingsdict[imagefile].append(rating)
    except Exception:
        ratingsdict[imagefile] = []
        ratingsdict[imagefile].append(rating)
    return ratingsdict

def get_avg_ratings(ratingsdict):
    """ Calcuate the average rating of each image. """
    for key in ratingsdict.keys():
        i = 0
        avg = 0
        for rating in ratingsdict[key]:
            avg += rating
            i += 1
        try:
            ratingsdict[key] = (avg/i-1)/9
        except Exception:
            pass

    return ratingsdict

def create_dataframe(files):
    """ Create the dataframe containing the info about the images and ratings. """
    ratingsdict = prepare_frame(files, use_full_folder='test')
    ratingframe = pd.DataFrame(list(ratingsdict.items()))
    return ratingframe

def prepare_image(file):
    """ Prepare image to train model with.
    
    The image will be resized to size 224x224 and preprocessed
    using keras and numpy functions.
    """
    print('prepare image', file)
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
    for file in files:
        file = glob(file)
    train = True
    name = "first_try"

    model = create_model()
    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[19:]:
        layer.trainable = True
    
    data = create_dataframe(files)
    print(data)

    if train:
        model = train_model(model, data, checkpoint=name, epochs=1, batch_size=20)
        model.save_weights(os.path.normpath(os.path.join(
            os.getcwd(), 'weights/' + name + '.h5')))
    else:
        model.load_weights("first_try.h5")

    predictions = test_on_batch(model, data.iloc[:,0])
    
    for i in range(len(predictions)):
        print(data.iloc[i,0], predictions[i]*9+1, data.iloc[i,1])

if __name__ == "__main__":
    main()
