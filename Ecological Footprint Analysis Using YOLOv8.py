# ======================================================
# ECO DETECTOR WITH FOOTPRINT % (CLASSROOM VERSION)
# ======================================================

!pip install ultralytics opencv-python --quiet

import cv2
import numpy as np
import base64
import time
from ultralytics import YOLO
from IPython.display import display, Javascript, HTML
from google.colab.output import eval_js

# ======================================================
# ECO RULES (UPDATED)
# ======================================================

eco_objects = [
    "potted plant",
    "book",
    "backpack",
    "bicycle",
    "bench",
    "bird",
    "cat",
    "dog",
    "person"      # 👕 assume clothing like jacket
]

non_eco_objects = [
    "laptop",
    "tv",
    "cell phone",
    "remote",
    "car",
    "motorcycle",
    "truck",
    "bus",
    "chair",
    "couch",
    "bottle"
]

# ======================================================
# WEBCAM SETUP
# ======================================================

def start_cam(width=640, height=480, q=0.7):

    display(HTML(f"""
    <div>
        <h3>🌍 ECOLOGICAL FOOTPRINT SYSTEM</h3>
        <img id="out" width="{width}" height="{height}">
    </div>
    """))

    display(Javascript(f"""
    window._w={width};
    window._h={height};
    window._q={q};

    window._video=document.createElement('video');
    window._video.setAttribute('autoplay','');
    window._video.setAttribute('playsinline','');

    window._canvas=document.createElement('canvas');
    window._canvas.width=_w;
    window._canvas.height=_h;
    window._ctx=_canvas.getContext('2d');

    window.startCam=async()=>{{
      window._stream=await navigator.mediaDevices.getUserMedia({{video:true}});
      window._video.srcObject=window._stream;
      await window._video.play();
    }}

    window.grabFrame=()=>{{
      window._ctx.drawImage(window._video,0,0,_w,_h);
      return window._canvas.toDataURL('image/jpeg',_q);
    }}

    startCam();
    """))

def grab_frame():
    data = eval_js("grabFrame()")
    if data is None:
        return None
    img_bytes = np.frombuffer(base64.b64decode(data.split(',')[1]), np.uint8)
    return cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

def show_frame(frame):
    _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    b64 = base64.b64encode(jpg).decode("utf-8")
    display(Javascript(f"""
        document.getElementById('out').src =
        'data:image/jpeg;base64,{b64}';
    """))

# ======================================================
# LOAD MODEL
# ======================================================

model = YOLO("yolov8n.pt")

print("📹 Starting Ecological Footprint Detector...")
start_cam()

# ======================================================
# MAIN LOOP
# ======================================================

try:
    target_fps = 8
    delay = 1.0 / target_fps

    while True:
        t0 = time.time()

        frame = grab_frame()
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)
        results = model(frame, imgsz=416, verbose=False)[0]

        eco_count = 0
        non_eco_count = 0

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            obj_name = model.names[cls]

            if obj_name in eco_objects:
                status = "ECO"
                color = (0,255,0)
                eco_count += 1

            elif obj_name in non_eco_objects:
                status = "NON-ECO"
                color = (0,0,255)
                non_eco_count += 1

            else:
                status = "NEUTRAL"
                color = (200,200,200)

            label = f"{obj_name} | {status}"

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 2)

        # ==============================
        # ECOLOGICAL CALCULATIONS
        # ==============================

        total = eco_count + non_eco_count

        if total > 0:
            eco_percentage = int((eco_count / total) * 100)
            footprint_percentage = int((non_eco_count / total) * 100)
        else:
            eco_percentage = 0
            footprint_percentage = 0

        # Display stats
        cv2.putText(frame, f"ECO OBJECTS: {eco_count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"NON-ECO OBJECTS: {non_eco_count}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(frame, f"ECO %: {eco_percentage}%", (10,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.putText(frame, f"ECOLOGICAL FOOTPRINT %: {footprint_percentage}%", (10,140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        show_frame(frame)

        dt = time.time() - t0
        if dt < delay:
            time.sleep(delay - dt)

except KeyboardInterrupt:
    print("🛑 Stopped.")
