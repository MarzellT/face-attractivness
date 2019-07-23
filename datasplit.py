import sys
import os
import random
from shutil import copyfile

choices = []
while len(choices)<int(sys.argv[1])-1:
    choice = random.choice(os.listdir(sys.argv[2]))
    if not (choice in choices):
        choices.append(choice)
        choice2 = random.choice(os.listdir(sys.argv[2]+'/'+choice))
        file = sys.argv[2]+choice+'/'+choice2
        print(file)
        copyfile(file, 'data/raten/' + choice+choice2)
