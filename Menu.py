from tkinter import *
from cv2 import *
import os
import faceRecognition as fr
import sys

count=0;
names={1:"Kangana",2:"Mohit",3:"Amit",4:"Animesh",5:"Purushottam",6:"Saurav"}
def Login():
    global login_screen;
    login_screen=Toplevel(main_screen);
    login_screen.geometry("1280x720");
    login_screen.title("Login");

    l1=Label(login_screen, text="Please enter details below to login");
    l1.config(font=('Courier',20,'bold'),fg="BLUE");
    l1.pack();
    Label(login_screen, text="").pack()

    global username_verify
    global password_verify

    username_verify = StringVar()
    password_verify = StringVar()
    global username_login_entry;
    global password_login_entry;
    Label(login_screen, text="Username * ",font=("Courier",10,"bold")).pack()
    username_login_entry = Entry(login_screen, textvariable=username_verify)
    username_login_entry.pack()
    Label(login_screen, text="").pack()
    Label(login_screen, text="Password * ",font=("Courier",10,"bold")).pack()
    password_login_entry = Entry(login_screen, textvariable=password_verify, show='*')
    password_login_entry.pack();
    Label(login_screen, text="").pack();
    Button(login_screen, text="Login", width=10, height=1, command=login_verify).pack()
    login_screen.mainloop();


def register():
    global username;
    global password;
    global username_entry;
    global password_entry;
    global register_screen;
    register_screen = Toplevel(main_screen)
    register_screen.title("Register")
    register_screen.geometry("1280x720")
    username = StringVar()
    password = StringVar()
    Label(register_screen, text="Please enter details below", bg="blue",font=("Courier",10,"bold")).pack()
    Label(register_screen, text="").pack()
    username_lable = Label(register_screen, text="Username * ",font=("Courier",15,"bold"))
    username_lable.pack()
    username_entry = Entry(register_screen, textvariable=username,width=20)
    username_entry.pack()
    password_lable = Label(register_screen, text="Password * ",font=("Courier",15,"bold"))
    password_lable.pack()
    password_entry = Entry(register_screen, textvariable=password, show='*')
    password_entry.pack()
    Label(register_screen, text="").pack()
    Button(register_screen, text="Register", width=10, height=1, bg="blue",command=register_user).pack()

def register_user():
    username_info = username.get()
    password_info = password.get()
    file = open(username_info, "w")
    file.write(username_info + "\n")
    file.write(password_info)
    file.close()
    username_entry.delete(0, END)
    password_entry.delete(0, END)
    Label(register_screen, text="Registration Success", fg="green", font=("calibri", 11)).pack();

def capture_face():
    global names
    path = r"C:\Users\Mohit\PycharmProjects\Myproject\venv\trainingImages";
    sub_dir=os.listdir(path)
    for x in sub_dir:
        pass
    x=int(x);
    count=x+1;
    cap = cv2.VideoCapture(0)
    name = face_of_user.get();
    os.chdir(path);

    os.mkdir(str(count));
    new_path = r"C:\Users\Mohit\PycharmProjects\Myproject\venv\trainingImages" + "\\" + str(count);
    os.chdir(new_path);
    names.update({count:name})

    c=0

    while True:
        ret, test_img = cap.read()
        if not ret:
            continue
        cv2.imwrite("frame%d.jpg" % c, test_img)  # save frame as JPG file
        c+= 1
        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('face detection Tutorial ', resized_img)
        if cv2.waitKey(10) == ord('q') or c == 60:  # wait until 'q' key is pressed
            break

    cap.release()
    cv2.destroyAllWindows

def Test_trainer():
    # This module takes images  stored in disk and performs face recognition
    test_img = cv2.imread(r'C:\Users\Mohit\PycharmProjects\Myproject\venv\Test_Images\Amit.jpg')  # test_img path
    faces_detected, gray_img = fr.faceDetection(test_img)
    print("faces_detected:", faces_detected)
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(r'C:\Users\Mohit\PycharmProjects\Myproject\venv\trainingData.yml')#use this to load training data for subsequent runs
    global names


    for face in faces_detected:
        (x, y, w, h) = face
        roi_gray = gray_img[y:y + h, x:x + h]
        label, confidence = face_recognizer.predict(roi_gray)  # predicting the label of given image
        print("confidence:", confidence)
        print("label:", label)
        fr.draw_rect(test_img, face)
        predicted_name = names[label]
        if (confidence < 37):  # If confidence more than 37 then don't print predicted face text on screen
            continue
        fr.put_text(test_img, predicted_name, x, y)

    resized_img = cv2.resize(test_img, (1000, 1000))
    cv2.imshow("face dtecetion tutorial", resized_img)
    cv2.waitKey(0)  # Waits indefinitely until a key is pressed
    cv2.destroyAllWindows


def Train_data():
    global names
    # This module takes images  stored in disk and performs face recognition
    test_img = cv2.imread(r'C:\Users\Mohit\PycharmProjects\Myproject\venv\Test_Images\Mohit.jpg')  # test_img path
    faces_detected, gray_img = fr.faceDetection(test_img)
    print("faces_detected:", faces_detected)

    # Comment belows lines when running this program second time.Since it saves training.yml file in directory
    faces, faceID = fr.labels_for_training_data(r'C:\Users\Mohit\PycharmProjects\Myproject\venv\trainingImages')
    face_recognizer = fr.train_classifier(faces, faceID)
    face_recognizer.write(r'C:\Users\Mohit\PycharmProjects\Myproject\venv\trainingData.yml')


    for face in faces_detected:
        (x, y, w, h) = face
        roi_gray = gray_img[y:y + h, x:x + h]
        label, confidence = face_recognizer.predict(roi_gray)  # predicting the label of given image
        print("confidence:", confidence)
        print("label:", label)
        fr.draw_rect(test_img, face)
        predicted_name = names[label]
        if (confidence > 37):  # If confidence more than 37 then don't print predicted face text on screen
            continue
        fr.put_text(test_img, predicted_name, x, y)

    resized_img = cv2.resize(test_img, (1000, 1000))
    cv2.imshow("face detecetion", resized_img)
    cv2.waitKey(0)  # Waits indefinitely until a key is pressed
    cv2.destroyAllWindows


def Recognize_face():
    # This module captures images via webcam and performs face recognition
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('trainingData.yml')  # Load saved training data

    name = {1: "Kangana",2:"Mohit",3:"Amit",4:"Animesh",5:"Purushottam",6:"Saurav"}

    cap = cv2.VideoCapture(0)

    while True:
        ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
        faces_detected, gray_img = fr.faceDetection(test_img)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('face detection Tutorial ', resized_img)
        cv2.waitKey(10)

        for face in faces_detected:
            (x, y, w, h) = face
            roi_gray = gray_img[y:y + w, x:x + h]
            label, confidence = face_recognizer.predict(roi_gray)  # predicting the label of given image
            print("confidence:", confidence)
            print("label:", label)
            fr.draw_rect(test_img, face)
            predicted_name = name[label]
            if confidence > 40:  # If confidence less than 37 then don't print predicted face text on screen
                fr.put_text(test_img, predicted_name, x, y)


        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('face recognition ', resized_img)
        if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
            break

    cap.release()
    cv2.destroyAllWindows


def login_success():
    global login_success_screen
    global face_of_user
    login_success_screen=Toplevel(login_screen)
    login_success_screen.geometry("1280x720")
    login_success_screen.title("Face_recognizer")
    face_of_user = StringVar()

    l1 = Label(login_success_screen, text="Face Recognizer");
    l1.config(font=('Courier', 30, 'bold'), fg="BLUE");
    l1.pack();
    Label(login_success_screen, text="Name of Face * ", font=("Courier", 10, "bold")).pack()
    username_login_entry = Entry(login_success_screen, textvariable=face_of_user)
    username_login_entry.pack()
    Label(text="").pack();
    Button(login_success_screen, text="ADD FACE", command=capture_face).pack()
    Label(text="").pack();
    Label(text="").pack();
    Button(login_success_screen, text="Train_data", command=Train_data).pack()
    Label(text="").pack();
    Label(text="").pack();

    Button(login_success_screen, text="Test_Trainer", command=Test_trainer).pack()
    Label(text="").pack();
    Label(text="").pack();

    Button(login_success_screen, text="Recognize_Face", command=Recognize_face).pack()
    Label(text="").pack();
    Label(text="").pack();



def password_not_recognized():
    global password_not_recog_screen
    password_not_recog_screen = Toplevel(login_screen)
    password_not_recog_screen.title("Success")
    password_not_recog_screen.geometry("150x100")
    Label(password_not_recog_screen, text="Invalid Password ").pack()
    Button(password_not_recog_screen, text="OK", command=delete_password_not_recognised).pack()


def user_not_found():
    global user_not_found_screen
    user_not_found_screen = Toplevel(login_screen)
    user_not_found_screen.title("Success")
    user_not_found_screen.geometry("150x100")
    Label(user_not_found_screen, text="User Not Found").pack()
    Button(user_not_found_screen, text="OK", command=delete_user_not_found_screen).pack()


def delete_login_success():
    login_success_screen.destroy()


def delete_password_not_recognised():
    password_not_recog_screen.destroy()


def delete_user_not_found_screen():
    user_not_found_screen.destroy()

def login_verify():
    username1 = username_verify.get()
    password1 = password_verify.get()
    username_login_entry.delete(0, END)
    password_login_entry.delete(0, END)
    list_of_files = os.listdir()
    if username1 in list_of_files:
        file1 = open(username1, "r")
        verify = file1.read().splitlines()
        if password1 in verify:
            login_success()
        else:
            password_not_recognized();
    else:
        user_not_found()


def main_account():
    global main_screen;
    main_screen=Tk();
    main_screen.geometry("1280x720");
    main_screen.title("Account Login");
    label1=Label(text="Login Or Register",width="1280", bg="grey",height="5");
    label1.config(font=("Courier", 44,'bold'),fg="BLUE");
    label1.pack();
    Label(text="").pack();
    login=Button(text="Login",width="100",height="5", command=Login);
    login.config(font=("Courier",12,'bold'))
    login.pack();
    Label(text="").pack();
    Register=Button(text="Register",width="100",height="5",command= register);
    Register.config(font=("Courier",12,'bold'));
    Register.pack();
    main_screen.mainloop()

main_account()
