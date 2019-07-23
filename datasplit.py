import sys
import os
import random
from shutil import copyfile

files = sys.argv[1:]
for i in range(0,len(files),3):
    choice = random.choice(os.listdir(files[i]))
    file = './'+files[i]+'/'+choice
    print(file)
    copyfile(file, 'data/raten/' + str(i) + choice)
