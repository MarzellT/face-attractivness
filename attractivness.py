import os
import sys
import keras_vggface
from keras_vggface.utils import preprocess_input
import pandas as pd
import mtcnn
from PIL import Image

files = sys.argv[1:]
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
        dirs[j] = dirs[j][:i-1]
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

# now actually get the average

for j in range(len(ratingslist)):
    i = 0
    avg = 0
    for rating in ratingslist[j]:
        avg += rating
        i += 1
    ratingslist[j] = avg/i

ratingframe = pd.DataFrame({'files': imagefiles, 'ratings': ratingslist})


# now the data is ready in form of
# filename  avgrating
# next setup model
# see what size the images have to be for input
# set output layer neuron count to 1 for regression
# put in some dense layers at the end to make use of feature extraction
# we need to preprocess the images (already imported a function)
# save the results with filename and net result
