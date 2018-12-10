How to run the program:
1) compile using python 3 blockmatch.py
2) make sure the video is in the same directory
3) to change frame save path change the variable frame_save_path to your desiredpath
4) to change output frames folder path change outputframe_save_path.
*you do not need to make this folder manually the program will create it if
not exist
5) to change diff frame output folder path change diff outputframe_save_path
*you do not need to make this folder manually the program will create it if
not exist
6) to change path to video change path_to_video
7) to change path to out put video change path_to_output_video
8) v0 and v1 are global variable for minimum SSD threshold. Change the number toany range for different video
9) withBoundaryOption is set to false initially. If you want the program to add 
boundaries to objects set this to true else the program will only do block
matching algorithm
10) use your texteditor to change all global variables mention above to your
desire settings. Not that boundary option can make performance slow.


