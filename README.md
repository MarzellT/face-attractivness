# Installation:nail_care:

## Download this repository      
- Click the green button "Clone or download" and then "Download ZIP".           
- Locate it in your downloads folder
⋅⋅1. Double click the folder
··2. Move folder `face-attractivness-master` to your desktop
- Windows User follow the stepts in the next section called **Windows**

## Windows:point_left:
### 1. Install Python3🐍
- Go to <https://www.python.org/downloads/windows/>   
- Download and install the latest version of Python3.   
··1.**Mark _☑️ Add Python 3.X to PATH_ in the installer.**    
··2. Click **Install**
- Proceed with step **Install dependencies**

### Install dependencies🤨
~~Open cmd (if you don't know how simply open the start menu and search for cmd).
Type `cd reposdirectory` (reposdirector is the path to where you have downloaded the files
for example `cd C:\Users\YourName\Desktop\face-attractivness-master` where YourName is 
the name of the current Windows user. This name is actually displayed when you open the cmd.)     
Now type `pip install -r requirements.txt`.~~   
😏      
Actually forget that stuff. 
😅
I implemented a much simpler feature.
- Run the `windows_setup` file from this folder.
- If a blue screen like this opens click **more Info** and then **run anyways**

## Advanced🤓 (You can skip that go to usage)
If you want you can also create a virtualenvironment specifically for this repository.      
To do so simply type `virtualenv venv` in the folder of this repository.      
Then you want to activate it by typing `source venv/bin/activate`.     
Now you can `pip install -r requirements.txt` in your virtualenvironment.    
To leave the virtualenvironment simply type `deactivate` or leave your terminal session.     

# Usage👩‍💻  
🚶‍♂️      
The following step is **not required** if you don't use a virtual environment which is also **not required**.
If you created a virtual environment (advanced installation section) activate it now.

## Windows
### What you need to do
- Double click face_rating.py         
- After finishing a new file will be in the folder called **ratings.csv**
- Send me this file💌     
- My email is <tobias.marzell@gmail.com>.     
You can also send me the file through any other service.🕊

### Advanced
You can also run this program through your terminal with arguments.    
Takes the directory of the images as arguments.     
Example: `python3 face_rating.py data/raten/*`.     
If you specify the location you can use complete or relative path.       

# Additional
You can create your own image folder using choices.py.     
This requries that images of differenct people are in differenct subfolders
because it will randomly select 1 picture of those folders. This ensures that
no person is selected twice.
`choices.py` does not use any dependecies.    
To run choices.py you must enter `choices.py arg1 args`.     
`arg1` is the amount of pictures you want and args are the folders in which your pictures are.    
 
 # Todo
 Assumption: I can assume that the score of one picture of one person generalises to all pictures of this person.    
 I can use this assumption to train on all images of that person to have more train data.     
 - Implement training based on this assumption.
