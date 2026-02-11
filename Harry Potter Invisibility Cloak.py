# ======================================================
# HARRY POTTER INVISIBILITY CLOAK (LIVE WEBCAM)
# ======================================================

import cv2, numpy as np, base64, time
from IPython.display import display, Javascript, HTML
from google.colab.output import eval_js

# ======================================================
# CAMERA SETUP
# ======================================================
def start_cam(W=640, H=480, q=0.7):
    display(HTML(f"""
    <div style="display:flex;gap:20px">
      <div>
        <h4>Webcam</h4>
        <img id="cam" width="{W}" height="{H}">
      </div>
      <div>
        <h4>Invisible Black Cloak</h4>
        <img id="out" width="{W}" height="{H}">
      </div>
    </div>
    """))

    display(Javascript(f"""
    window._w={W}; window._h={H}; window._q={q};
    window._video=document.createElement('video');
    window._video.setAttribute('autoplay','');
    window._video.setAttribute('playsinline','');

    window._canvas=document.createElement('canvas');
    _canvas.width=_w; _canvas.height=_h;
    window._ctx=_canvas.getContext('2d');

    window.startCam=async()=>{{
      window._stream=await navigator.mediaDevices.getUserMedia({{video:true}});
      _video.srcObject=_stream;
      await _video.play();
    }}

    window.grab=()=>{{
      _ctx.drawImage(_video,0,0,_w,_h);
      return _canvas.toDataURL('image/jpeg',_q);
    }}

    startCam();
    """))

def grab_frame():
    data = eval_js("grab()")
    img = base64.b64decode(data.split(',')[1])
    return cv2.imdecode(np.frombuffer(img,np.uint8),cv2.IMREAD_COLOR)

def show_frames(cam, out):
    def enc(img):
        _, j = cv2.imencode(".jpg", img)
        return base64.b64encode(j).decode()
    display(Javascript(f"""
    document.getElementById('cam').src='data:image/jpeg;base64,{enc(cam)}';
    document.getElementById('out').src='data:image/jpeg;base64,{enc(out)}';
    """))

# ======================================================
# CAPTURE BACKGROUND
# ======================================================
print("⏳ Capturing background... stay out of camera view")
start_cam()

bg_frames = []
for i in range(30):
    frame = cv2.flip(grab_frame(), 1)
    bg_frames.append(frame)
    time.sleep(0.1)

background = np.median(bg_frames, axis=0).astype(np.uint8)
print("✅ Background captured")

# ======================================================
# INVISIBILITY CLOAK FUNCTION
# ======================================================
def invisible_cloak(frame, background):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect dark (black) regions
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])  # allow dark shades
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Smooth edges
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask_inv = cv2.bitwise_not(mask)

    # Combine foreground and background
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bg = cv2.bitwise_and(background, background, mask=mask)
    result = cv2.addWeighted(fg, 1, bg, 1, 0)

    return result

# ======================================================
# RUN LOOP
# ======================================================
print("🎩 Cloak effect running. Press Ctrl+C to stop.")
while True:
    frame = cv2.flip(grab_frame(), 1)
    output = invisible_cloak(frame, background)
    show_frames(frame, output)
    time.sleep(0.05)
