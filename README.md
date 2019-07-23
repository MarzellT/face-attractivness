Usage:      
      
The first thing you need to do is to clone this repository.        
You can do that by either using the webclient or through your teminal by typing:       
`git@github.com:MarzellT/face-attractivness.git`      

The next step is to go to your terminal and type `pip install -r requirements.txt`.     
If you want you can also create a virtualenvironment specifically for this repository.      
To do so simply type `virtualenv venv` in the folder of this repository.      
Then you want to activate it by typing `source venv/bin/activate`.     
Now you can `pip install -r requirements.txt` in your virtualenvironment.    
To leave the virtualenvironment simply type `deactivate` or leave your terminal session.     

To use the face rater activate your virtualenvironment if you created one.      
Then start it with `python3 face_rating.py`.     
You can also run this script using arguments.     
The only argument available is where your image folder is located at.    
If you specify the location you can use complete or relative path.       

You can create your own image folder using choices.py.     
`choices.py` does not use any dependecies.    
To run choices.py you must enter `choices.py arg1 args`.     
`arg1` is the amount of pictures you want and args are the folders in which your pictures are.    
`choices.py` will select 1 random picture out of every folder.
