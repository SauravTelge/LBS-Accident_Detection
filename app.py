from flask import Flask, render_template, request, redirect, jsonify,Response
from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import os
from random import randrange
from sklearn.preprocessing import LabelBinarizer
# from google.colab.patches import cv2_imshow
from keras.preprocessing import image
import numpy as np
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from PIL import Image as im
UPLOAD_FOLDER = "/tmp/example_clips"
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
global path
path = '' 
img_dir = '/tmp/output_images/'
model = './assets/model/activity.model'
label_bin = './assets/model/lb.pickle'
input_dir = './static/example_clips/testeer.mp4'
size = 128
model = load_model(model)
lb = pickle.loads(open(label_bin, "rb").read())

def gen_frames():  
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    Q = deque(maxlen=size)

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    global path
    vs = cv2.VideoCapture(path)
    writer = None
    (W, H) = (None, None)
    counter = 0
    def getFrame(sec):
        vs.set(cv2.CAP_PROP_POS_MSEC,sec*100)
        (grabbed, frame) = vs.read()
        return (grabbed,frame)
# loop over frames from the video file stream
    secs = 0
    print("hi0")
    while True:
        # read the next frame from the file
        secs = secs + 1
        (grabbed,frame) = getFrame(secs)
        print("hi1")
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # clone the output frame, then convert it from BGR to RGB
        # ordering, resize the frame to a fixed 224x224, and then
        # perform mean subtraction
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype("float32")
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        # make predictions on the frame and then update the predictions
        # queue
        x = image.img_to_array(frame)
        # plt.imshow(x/255.)
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        print(preds)
        Q.append(preds)

        # perform prediction averaging over the current history of
        # previous predictions
        i = np.argmax(preds)
        print("i")
        print(i)
        # draw the activity on the output frame
      
        

        # capture all the frame of accident and save it in output_images folder
        if (i == 0):
            label = "accident"
            counter = counter + 1
            print("hi3")
            alert = "warning:{}".format(label)
            cv2.putText(output, alert, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,255,255), 5)
            irand = randrange(0, 1000)
            # write the output image to disk
            filename = "{}.png".format(irand)
            p = os.path.sep.join([img_dir, filename])
            cv2.imwrite(p, output)

        if (i == 2):
            label = "accident"
            counter = counter + 1
            alert = "warning:{}".format(label)
            cv2.putText(output, alert, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,255,255), 5)
            irand = randrange(0, 1000)
            # write the output image to disk
            filename = "{}.png".format(irand)
            p = os.path.sep.join([img_dir, filename])
            """ cv2.imwrite(p, output) """
        
        print("printing output before mailing it")
        # cv2.imshow("output",output)

        # sending the mail
        if(counter>5):
            print("printing output before mailing it")
            # cv2.imshow("output",output)
            strFrom = 'lbtestingacc@gmail.com'
            strTo = '2018.saurav.telge@ves.ac.in'
            msgRoot = MIMEMultipart('related')
            msgRoot['Subject'] = 'Accident occured at this  location. send help'
            msgRoot['From'] = strFrom
            msgRoot['To'] = strTo
            msgRoot.preamble = '====================================================='
            msgAlternative = MIMEMultipart('alternative')
            msgRoot.attach(msgAlternative)
            msgText = MIMEText('<img src="cid:image1"><br>', 'html')
            msgAlternative.attach(msgText)
            data = im.fromarray(output)
            data.save('accident.png')
            fp = open('accident.png', 'rb')
            msgImage = MIMEImage(fp.read())
            fp.close()
            msgImage.add_header('Content-ID', '<image1>')
            msgRoot.attach(msgImage)
            smtp=smtplib.SMTP("smtp.gmail.com", 587)
            smtp.ehlo()
            smtp.starttls()
            smtp.login("lbtestingacc@gmail.com", "sauravtelge#1")
            smtp.sendmail(strFrom, strTo, msgRoot.as_string())
            smtp.quit()
            break
        ret, buffer = cv2.imencode('.jpg', output)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    print("[INFO] cleaning up...")
 
    path = ""
    vs.release()
@app.route("/upload",methods=["GET", "POST"])
def upload():
    f = request.files["file"]
    f.filename=f.filename.replace(" ","_")
    f.save(os.path.join(app.config["UPLOAD_FOLDER"], f.filename))
    global path
    path = f'/tmp/{f.filename}'
    return home()

@app.route("/")
def home():

    
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)