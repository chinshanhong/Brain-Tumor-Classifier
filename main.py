#Import necessary library to create GUI, process GUI and load machine learning model
import cv2
import numpy as np
import tkinter.filedialog
import tensorflow as tf
import tkinter
from tkinter import *
from tkinter.ttk import *
from tkinter import ttk
from tkinter.filedialog import askopenfile
from PIL import ImageTk, Image

#Create a main window for the GUI
root = Tk()
root.title("Brain Tumor Classifier")

#Set the height = 600 and width = 500
root.geometry('600x500')
#Fix the size of the main window
root.resizable(0, 0)

#Create and display the name of our program
program_name = Label(root, text="Brain Tumor Classifier", font=("Arial", 25))
program_name.pack()

#Set the background colour of our image frame to black
s = ttk.Style()
s.configure("ImageFrame.TFrame", background='black')

#Create and display the image frame with height = 300 and width = 300
image_frame = Frame(root, height=300, width=300, style='ImageFrame.TFrame')
image_frame.pack()

#Create a result frame to hold the prediction result
result_frame = Frame(root)
result_frame.pack()

#A function to read the input image selected by the user and
# classify the tumor in the image
def predict_image():
    global img
    clear_image_and_result()
    file_types = [('Jpg Files', '*.jpg')]
    filename = tkinter.filedialog.askopenfilename(filetypes=file_types)
    img = Image.open(filename)
    img_resized = img.resize((300, 300))
    img = ImageTk.PhotoImage(img_resized)
    brain_tumor_image = Label(image_frame, image=img)
    brain_tumor_image.pack()
    display_prediction_result(filename)

#A function to clear the image and result widget
def clear_image_and_result():
    for image in image_frame.winfo_children():
        image.destroy()
    for result in result_frame.winfo_children():
        result.destroy()

#A function to display the prediction result
def display_prediction_result(filename):
    img_for_predict = cv2.imread(filename)
    img_for_predict_resized = cv2.resize(img_for_predict, (150, 150))
    xception_model = tf.keras.models.load_model(
        "C:/Users/shanh/Desktop/UM Files/Year 2 Sem 2/WIX3001 Soft Computing/xception.h5")
    img_for_predict_resized = (np.expand_dims(img_for_predict_resized, 0))
    x_predict = []
    x_predict.append(img_for_predict_resized)
    x_predict = np.array(img_for_predict_resized)
    prediction = xception_model.predict(x_predict)
    predictied_class_index = np.argmax(prediction, axis=1)
    class_name = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
    prediction_result = Label(result_frame, text= "Result: " + class_name[predictied_class_index[0]] + "", font=('Arial', 20))
    prediction_result.pack()

#Create a button frame to hold all the buttons in the GUI
button_frame = Frame(root)
button_frame.pack()

#Create an upload button
upload_button = Button(button_frame, text="Predict Image", command=lambda: predict_image())
upload_button.pack(pady=15, side=tkinter.LEFT)

#Create an clear button
clear_button = Button(button_frame, text="Clear", command=lambda: clear_image_and_result())
clear_button.pack(pady=20, side=tkinter.LEFT)

root.mainloop()
