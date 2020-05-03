# Facial Recognition with Python

This is a simple program in python that uses face_recognition module to recognize faces in images.
Opencv module is used for making rectangles and text area around the face.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install face_recognition and cv packages.

```bash
pip install face_recognition
```
```bash
pip install opencv-python
```
## Usage
The known_faces directory should contain individual images of different people in separate folders so that the program can first encode those faces and store their names in a list.

The unknown_faces directory should contain the images that need to be processed for identification.

Once done, run the program and it should output seperate windows each with one image that identifies the people in those images.
