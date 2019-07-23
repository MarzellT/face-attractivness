import sys
import keras_vggface
import mtcnn
from PIL import Image

files = sys.argv[1:]
for f in files:
    image = None
    try:
        image = Image.open(f + '/0001_01.jpg')
    except Exception:
        try:
            image = Image.open(f + '/0002_02.jpg')
        except Exception:
            print(f)
    if image != None:
        print(image)
        image.resize((300,300))
        image.show()
