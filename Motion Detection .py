import cv2

import numpy as np

import time

import base64

from IPython.display import display, Audio

from google.colab.output import eval_js

 

# ----------------------------------

# Webcam capture (Colab safe)

# ----------------------------------

def capture_frame():

    js_code = """

    async function capture() {

      const video = document.createElement('video');

      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(video);

      video.srcObject = stream;

      await video.play();

      await new Promise(resolve => setTimeout(resolve, 500));

 

      const canvas = document.createElement('canvas');

      canvas.width = video.videoWidth;

      canvas.height = video.videoHeight;

      canvas.getContext('2d').drawImage(video, 0, 0);

 

      const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

      stream.getTracks().forEach(track => track.stop());

      video.remove();

      return dataUrl;

    }

    capture();

    """

    data_url = eval_js(js_code)

    # Remove the header "data:image/jpeg;base64,"

    img_str = data_url.split(',')[1]

    # Decode base64 to bytes

    img_bytes = base64.b64decode(img_str)

    # Convert bytes to numpy array

    img_array = np.frombuffer(img_bytes, dtype=np.uint8)

    # Decode numpy array to OpenCV image

    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return frame

 

# ----------------------------------

# Capture background frame

# ----------------------------------

print("📸 Capturing background frame...")

print("⚠️  PLEASE MOVE OUT OF CAMERA VIEW")

time.sleep(3)

 

background = capture_frame()

background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

 

print("✅ Background captured")

 

# ----------------------------------

# Monitoring loop

# ----------------------------------

ALARM_THRESHOLD = 50000

FRAME_DELAY = 1

 

print("🔍 Monitoring started...")

 

while True:

    frame = capture_frame()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (21, 21), 0)

 

    diff = cv2.absdiff(background_gray, gray)

    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    changed_pixels = cv2.countNonZero(thresh)

    print("Changed pixels:", changed_pixels)

 

    if changed_pixels > ALARM_THRESHOLD:

        print("🚨🚨 INTRUSION DETECTED 🚨🚨")

        display(Audio(

            url="https://www.soundjay.com/buttons/sounds/beep-07.mp3",

            autoplay=True

        ))

        break

 

    time.sleep(FRAME_DELAY)
