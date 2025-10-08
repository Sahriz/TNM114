# Overview
This is a project that I made as part of the course *Artificial Intelligence for Interactive Media - TNM114*. It is currently a WIP, but when finnished
will consist trained convolutional neural networks which are trained on my own data to identify hand gestures using a web cam which will be used in a video game to control a swarm of characters. The project combines
both python and gd script, where the python part of the system uses opencv and tensorflow to read the data and classify it. After which the godot part of the system listens through a web socket for the user inputs 
and steers the flock depending on the classified gestures. 

# Data
The current data is temporary and only meant to test the tensorflow framework. After the whole system is up and running, the dataset will be expanded and uploaded to Kaggle for others to use as well. If there is a demand,
I might just add gestures that others want to the dataset when I find the time and drive to do so. The plan is to include multiple people into the dataset and with mulitple settings to make sure that the dataset doesn't
easily overfit in a well made ML system.
