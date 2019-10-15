import os
import datetime
import glob
import argparse
import sys
import random
import keras
import keras_vggface
from pathlib import Path, PureWindowsPath
from keras.preprocessing import image
from keras.layers import Dense,GlobalAveragePooling2D, Activation
from keras.models import Model, model_from_json
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import keras.backend as K
from glob import glob
import numpy as np
import pandas as pd
import mtcnn
from PIL import Image, ImageDraw, ImageFont

def prepare_frame(files, use_actual_path=True, use_full_folder=None, endearly=None):
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
    if endearly:
        # implent raise error endearly requires use_full_folder
        pass
    else:
        endearly = False
    if use_full_folder:
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
            # setup path
            # i figured this weird way is needed for this to work in windows (not tested though)
            filename = dirs[j][i:]
            dirs[j] = PureWindowsPath(dirs[j][:i-1])
            dirs[j] = Path(dirs[j])
            dirname = os.path.basename(dirs[j])
            basefoldername = os.path.dirname(os.path.dirname(dirs[j]))
            basefoldername = os.path.join(basefoldername, 'test')

            dirrating = ratings[j]
            curdir = os.path.join(basefoldername, dirname)

            fncnt.append(curdir)
            if use_full_folder:
                try:
                    # here we take every image of the folder and assign the rating of
                    # the one rated picture to all based on assumption described in the
                    # README file
                    for num, filename in enumerate(os.listdir(curdir)):
                        imagefile = os.path.join(curdir, filename)
                        ratingsdict = add_to_dict(imagefile, dirrating, ratingsdict)
                        if num > endearly and not (endearly == False):
                            break
                except Exception as e:
                    print(e, 'at:', os.path.join(basefoldername, curdir))
            else:
                imagefile = os.path.join(curdir, filename)
                ratingsdict = add_to_dict(imagefile, dirrating, ratingsdict)

    ratingsdict = get_avg_ratings(ratingsdict)
    print('FOUND', len(ratingsdict.keys()), 'IMAGES')
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

def create_dataframe(files, use_full_folder=False, endearly=None):
    """ Create the dataframe containing the info about the images and ratings. """
    ratingsdict = prepare_frame(files, use_full_folder=use_full_folder, endearly=endearly)
    ratingframe = pd.DataFrame(list(ratingsdict.items()))
    return ratingframe

def prepare_image(file, target_size=(224,224)):
    """ Prepare image to train model with.
    
    The image will be resized to size 224x224 and preprocessed
    using keras and numpy functions.
    """
    try:
        prepared = image.load_img(file, target_size=target_size)
        prepared = image.img_to_array(prepared)
        prepared = np.expand_dims(prepared, axis=0)
        prepared = preprocess_input(prepared)
    except Exception:
        raise ValueError
    return prepared[0]

def create_model(neurons=224, target_size=(224,224), activation='sigmoid'):
    """ Create the transfer learning model. """
    base_model = VGGFace(include_top=False, input_shape=(target_size[0], target_size[1], 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(neurons,activation='relu')(x)
    #final layer
    preds = Dense(1, activation=activation)(x)

    model = Model(inputs=base_model.input,outputs=preds)
    return model

def train_model(model, weights, modelname, files, targets, epochs=20, batch_size=32, save_best_only=None):
    """ Return trained model and train history.

    Trains a model for given data and optionally epochs. 
    model: kerasmodel to train on
    weights: filename of the weights to load and train on
    modelname: name to save the folder of the weights
    files: image file names
    targets: rating target
    epochs: number of epochs
    batch_size: keras batch_size
    save_best_only: if true only the best weights will be saved.
    """
    # Load training images and tagets into numpy array obejct.
    print('LOADING', len(files), 'IMAGES')
    print(files)
    images = []
    for i in range(len(files)):
        images.append(prepare_image(files[i]))
        if i % int((len(files)/20) + 1) == 0:
            print(i, 'IMAGES LOADED')
    print('LOADING FINISHED WITH', len(images), 'IMAGES')
    images = np.array(images)
    targets = np.array(targets)

    model.compile(optimizer='Adam', loss='mse')
    if weights:
        model.load_weights(weights)
    modelpath = os.path.join('logs', modelname)
    filepath = os.path.join(modelpath, 'epoch{epoch:02d}.hdf5')
    checkpoints = []
    checkpoints.append(ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=save_best_only,
        save_weights_only=True, mode='auto', period=1))
    checkpoints.append(TensorBoard(log_dir=modelpath, histogram_freq=0, batch_size=batch_size,
        write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
        embeddings_metadata=None, embeddings_data=None, update_freq='epoch'))

    model.fit(images, targets, validation_split=0.25, callbacks=checkpoints, epochs=epochs, batch_size=batch_size, shuffle=True)
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
    """ Test model on files passed as list. """
    batch = []
    for file in files:
        batch.append(prepare_image(file))
    batch = np.array(batch)
    preds = model.predict_on_batch(batch)
    return preds

def visualize_result(file, prediction, ground_truth=None, color=(255,255,255),
        save=False, fontsize=30):
    """ Save the file with its prediction as an image.

    file: the pathname to the image
    prediction: the prediction of the image
    ground_truth: if given also add the ground truth
    color: tuple of rgb color
    """
    image = Image.open(file)
    drawtext = ImageDraw.Draw(image)
    font = ImageFont.truetype("NotoMono-Regular.ttf", fontsize)
    if ground_truth:
        drawtext.text((0,0), str(round(prediction[0], 2)) + "\n" + str(ground_truth), color, font=font)
    else:
        drawtext.text((0,0), str(round(prediction[0], 2)), color, font=font)
    if save:
        savepath = os.path.join('predicted', os.path.basename(file))
        image.save(savepath)
    return image

def main():
    #try:
    parser = argparse.ArgumentParser(description='files need to be the rating files')
    parser.add_argument("-f", "--files", nargs='*', required=True)
    parser.add_argument("-n", "--number", type=int, nargs=1)
    parser.add_argument("--entirefolder", action='store_true')
    parser.add_argument("-t", "--train", action='store_true')
    parser.add_argument("-e", "--epochs", type=int, nargs=1)
    parser.add_argument("-w", "--weights", type=str, nargs=1)
    parser.add_argument("-b", "--batchsize", type=int, nargs=1)
    args = parser.parse_args(sys.argv[1:])
    modelname = datetime.datetime.now().strftime("%d%m%Y%H%M")

    try:
        endearly = args.number[0]
        entire = True
    except Exception:
        endearly = None
        if args.entirefolder:
            entire = True
        else:
            entire = False
    if args.batchsize:
        batch_size = args.batchsize[0]
    else:
        batch_size = 20

    files = args.files

    if args.train and args.weights:
        train = args.train
        weights = args.weights[0]
    elif args.train:
        train = args.train
        weights = None
    elif args.weights:
        weights = args.weights[0]
        train = False
    else:
        print('-t or -w is required')

    model = create_model()

    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[19:]:
        layer.trainable = True

    if train:
        data = create_dataframe(files, entire, endearly)
        files = data.iloc[:,0]
        targets = data.iloc[:,1]
        print('MODE: TRAIN')
        epochs = args.epochs[0]
        print('TRAINING FOR', epochs, 'EPOCHS')
        model = train_model(model, weights=weights, modelname=modelname, files=files, targets=targets, epochs=epochs, batch_size=batch_size)
    else:
        print('MODE: INFERENCE')
        print('LOADING WEIGHTS:', weights)
        model.load_weights(weights)
        for i in range(len(files)):
            if os.path.isdir(files[i]):
                files[i] = os.path.join(files[i], random.choice(os.listdir(files[i])))
        print('PREDICTING')
        predictions = test_on_batch(model, files)
        
        for i in range(len(predictions)):
            visualize_result(files[i], predictions[i]*9+1, color=(255,255,0), save=True)

if __name__ == "__main__":
    main()
