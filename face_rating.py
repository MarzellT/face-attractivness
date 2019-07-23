import sys
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

class Window:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Facebewerterer")
        self.window.geometry("500x500")

        files = sys.argv[1:]
        images = []
        paths = []
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
        
        self.endflag = False
        
        self.window.bind('<Return>', self.call_rating_call_back)
        self.window.mainloop()


    def wrong_number_call_back(self):
        messagebox.showerror( "Wrong value error", "Geb eine Zahl zwischen 1 und 10 ein.")

    def nan_call_back(self):
        messagebox.showerror( "Not a number error", "Geb eine Zahl zwischen 1 und 10 ein.")

    def finish_call_back(self):
        if not self.endflag:
            dank_sagung = ("Vielen Dank für deine Hilfe, du bist großartig!\n" +\
            "Als letzten Schritt bitte ich dich die Datei 'ratings.csv' noch an mich zu schicken.\n" +\
            "Nochmal vielen Dank für deine Zeit.")
            messagebox.showinfo("Vielen Dank!", dank_sagung)
            with open('ratings.csv', 'w') as file:
                file.write(self.ratings)
            self.rating_button.config(state=tk.DISABLED)
            self.finish_button.config(state=tk.DISABLED)
            self.rating_field.config(state=tk.DISABLED)
            self.endflag = True

    def rating_call_back(self):
        if not self.endflag:
            rating = self.rating_field.get()
            if self.current_image >= len(self.photo_images):
                self.finish_call_back()
            else:
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
        self.rating_call_back()

def main():
    w = Window()

if __name__ == "__main__":
    main()
