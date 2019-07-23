import os
import sys
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

""" face_rating.py

This program is used to rate faces in images.
It will save a csv of first element = file and second elemt = rating 
after successful run or early aboard.

This program can be run with arguments or without.
If run without arguments it will search for images in the directory
of ./data/raten/ . Else you can specify the location of your images.
"""

__author__ = "Tobias Marzell"
__credits__ = "Credits to anyone helping me collecting data."
__email__ = "tobias.marzell@gmail.com"

class Window:
    """ face_rating window object.

    Contains all the information of the window.
    This includes the basic window structure aswell as
    the rating logic.
    """

    def __init__(self):
        """ Create the window and setup variables. """
        self.window = tk.Tk()
        self.window.title("Facebewerterer")
        self.window.geometry("500x500")

        images = []
        paths = []

        # Choose whether run with arguments or not.
        if len(sys.argv) > 1:
            files = sys.argv[1:]
        else:
            dir = os.path.normcase('data/raten/')
            files = os.listdir(dir)
            for i in range(len(files)):
                files[i] = dir + files[i]

        
        # Load images.
        for f in files:
            image = None
            try:
                path = (f)
                image = Image.open(path)
            except Exception:
                print(f)
            if image != None:
                paths.append(path)
                image = image.resize((300,300))
                images.append(image)


        # Create tk usable images.
        photo_images = []
        for image in images:
            photo_image = ImageTk.PhotoImage(image)
            photo_images.append(photo_image)
        self.photo_images = photo_images
        
        self.paths = paths
        self.current_image = 0
        self.ratings = ''

        self.rating_button = tk.Button(self.window, text="Eingabe", command=self.rating_call_back)
        self.finish_button = tk.Button(self.window, text="Beenden", command=self.finish_call_back)
        self.rating_field = tk.Entry(self.window)
        self.panel = tk.Label(self.window, image = photo_images[self.current_image])
        self.progress = tk.Label(self.window, text="Bild " + str(self.current_image + 1) + " von " + str(len(self.photo_images)) + ".")
        
        self.progress.pack()
        self.panel.pack() 
        self.rating_field.pack()
        self.rating_button.pack()
        self.finish_button.pack()
        
        # If endflag window gets inactive and csv is created.
        self.endflag = False
        
        self.window.bind('<Return>', self.call_rating_call_back)
        self.window.mainloop()


    def wrong_number_call_back(self):
        """ Create errorbox if number is wrong. """
        messagebox.showerror( "Wrong value error", "Geb eine Zahl zwischen 1 und 10 ein.")

    def nan_call_back(self):
        """ Create errorbox if not a number. """
        messagebox.showerror( "Not a number error", "Geb eine Zahl zwischen 1 und 10 ein.")

    def finish_call_back(self):
        """ Make window inactive and show that program is finished. """
        if not self.endflag:

            dank_sagung = ("Vielen Dank für deine Hilfe, du bist großartig!\n" +\
            "Als letzten Schritt bitte ich dich die Datei 'ratings.csv' noch an mich zu schicken.\n" +\
            "Nochmal vielen Dank für deine Zeit.")
            messagebox.showinfo("Vielen Dank!", dank_sagung)

            # save csv
            with open('ratings.csv', 'w') as file:
                file.write(self.ratings)

            # disable window
            self.rating_button.config(state=tk.DISABLED)
            self.finish_button.config(state=tk.DISABLED)
            self.rating_field.config(state=tk.DISABLED)
            self.endflag = True

    def rating_call_back(self):
        """ Add rating to string and get next picture. """
        if not self.endflag:
            rating = self.rating_field.get()
            # if last picture end programm
            if self.current_image >= (len(self.photo_images) - 1):
                self.finish_call_back()
            else:
                # check for correct entry
                if rating.isnumeric():
                    if int(rating) > 0 and int(rating) < 11:
                        self.ratings += self.paths[self.current_image] + ';' + rating + '\n'            
                        self.current_image += 1
                        self.panel.config(image = self.photo_images[self.current_image])
                        self.progress.config(text=("Bild " + str(self.current_image + 1) + " von " + str(len(self.photo_images)) + "."))
                        self.rating_field
                        self.rating_field.delete(0,'end')
                    else:
                        self.wrong_number_call_back()
                else:
                    self.nan_call_back()

    def call_rating_call_back(self, event):
        """ Wrapper for using <'Return'> """
        self.rating_call_back()

def main():
    """ Create window. """
    w = Window()

if __name__ == "__main__":
    main()
