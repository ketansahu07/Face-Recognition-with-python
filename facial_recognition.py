## IMPORTING MODULES
import face_recognition
import os
import cv2
# from google.colab.patches import cv2_imshow
# from matplotlib import pyplot as plt  		# in case opencv doesn't work

## DEFINING CONSTANTS
# path to the known_faces directory, here it is in the same folder as the program
KNOWN_FACES_DIR = "known_faces"
# path to the unknown_faces dir
UNKNOWN_FACES_DIR = "unknown_faces"
# lower the tolerance higher is the accuracy but again too low may not even recognize
# and too high may recognize all the faces as identical
TOLERANCE = 0.5   
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"   # hog for CPU, cnn for GPU

## LOADING KNOWN FACES
print("Loading known faces...")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
	for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
    	image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
    	encoding = face_recognition.face_encodings(image)[0]    #encoding at the zeroth index
    	known_faces.append(encoding)
    	known_names.append(name)

## ITERATE THROUGH UNKNOWN FACES
print("Processing unknown faces...")

for filename in os.listdir(UNKNOWN_FACES_DIR):
 	print(f"We are in {filename}")
	image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
	locations = face_recognition.face_locations(image, model=MODEL)   # finds the coordinates of the faces in the image
	encodings = face_recognition.face_encodings(image, locations)     # encode all those faces
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	for face_encoding, face_location in zip(encodings, locations):
    	results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
    	match = None
    	if True in results:
      		match = known_names[results.index(True)]
      		print(f"Match found: {match}")

      		top_left = (face_location[3], face_location[0])       # coordinate of rectangel to form around face
      		bottom_right = (face_location[1], face_location[2])   # coordinate of ractangel (since it takes top-left and bottom right to make a rectangle)

      		color = [0, 255, 0]     #color of the frame around the face

      		cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

      		top_left = (face_location[3], face_location[2])
      		bottom_right = (face_location[1], face_location[2]+22)    # bottom solid rectangel to display names of the person 

      		cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

      		# this puts the name there in the bottom solid rectangle
      		cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)

	#plt.imshow(image)
	#cv2_imshow(image)            # for google colab
	cv2.imshow(filename, image)   # this charshes the google colab kernal
	cv2.waitKey(10000)
	cv2.destroyWindow(filename)