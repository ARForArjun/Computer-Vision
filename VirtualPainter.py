import cv2, numpy as np, base64, time
from IPython.display import display, Javascript, HTML
from google.colab.output import eval_js

# ---------- Browser webcam ----------
def start_cam(width=640, height=480, jpeg_quality=0.6):
    display(HTML(f"""
      <div style="font-family:monospace;margin:6px 0;">
        Painter running... Stop the cell to end.
        <br/>Show a RED object to move the brush (centroid of red).
        <br/>Touch side rectangles with the red centroid to pick color. Touch BLACK to clear.
      </div>
      <img id="out" width="{width}" height="{height}" />
    """))
    display(Javascript(f"""
      window._w = {width};
      window._h = {height};
      window._q = {jpeg_quality};

      if (!window._cam_started) {{
        window._cam_started = true;

        window._video = document.createElement('video');
        window._video.setAttribute('autoplay','');
        window._video.setAttribute('playsinline','');

        window._canvas = document.createElement('canvas');
        window._canvas.width = window._w;
        window._canvas.height = window._h;
        window._ctx = window._canvas.getContext('2d');

        window.startCam = async function() {{
          if (window._stream) return "already";
          window._stream = await navigator.mediaDevices.getUserMedia({{video:true}});
          window._video.srcObject = window._stream;
          await window._video.play();
          return "started";
        }}

        window.grabFrame = async function() {{
          if (!window._stream) return null;
          window._ctx.drawImage(window._video, 0, 0, window._w, window._h);
          return window._canvas.toDataURL('image/jpeg', window._q);
        }}

        window.stopCam = function() {{
          if (window._stream) {{
            window._stream.getTracks().forEach(t => t.stop());
            window._stream = null;
          }}
        }}

        startCam();
      }}
    """))

def stop_cam():
    display(Javascript("if (window.stopCam) window.stopCam();"))

def grab_bgr():
    data = eval_js("grabFrame()")
    if data is None:
        return None
    img_bytes = base64.b64decode(data.split(',')[1])
    bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    return bgr

def show_in_browser(bgr):
    ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return
    b64 = base64.b64encode(jpg).decode("utf-8")
    display(Javascript(f"document.getElementById('out').src='data:image/jpeg;base64,{b64}';"))

# ---------- Red centroid ----------
def red_centroid(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # red hue wrap
    lower1 = np.array([0, 80, 50], np.uint8)
    upper1 = np.array([10, 255, 255], np.uint8)
    lower2 = np.array([170, 80, 50], np.uint8)
    upper2 = np.array([180, 255, 255], np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # largest blob centroid
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None, mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    blob = (labels == idx).astype(np.uint8) * 255
    M = cv2.moments(blob)
    if M["m00"] == 0:
        return None, blob
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), blob

# ---------- UI rectangles (right side) ----------
def make_buttons(W, H):
    pad = 10
    bw = 80
    bh = 60
    x1 = W - bw - pad
    # Red, Green, Blue, Black (clear)
    btns = [
        ("R", (0, 0, 255), (x1, pad, W - pad, pad + bh)),
        ("G", (0, 255, 0), (x1, pad + (bh + pad)*1, W - pad, pad + (bh + pad)*1 + bh)),
        ("B", (255, 0, 0), (x1, pad + (bh + pad)*2, W - pad, pad + (bh + pad)*2 + bh)),
        ("CLR", (0, 0, 0), (x1, pad + (bh + pad)*3, W - pad, pad + (bh + pad)*3 + bh)),
    ]
    return btns

def draw_buttons(frame, btns, current_color):
    for name, bgr, (x1,y1,x2,y2) in btns:
        cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, -1)
        # outline
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)
        # label
        txt = name
        txt_color = (255,255,255) if bgr == (0,0,0) else (0,0,0)
        cv2.putText(frame, txt, (x1+10, y2-18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2, cv2.LINE_AA)
    # show current color swatch at bottom-left
    cv2.rectangle(frame, (10, H-50), (80, H-10), current_color, -1)
    cv2.rectangle(frame, (10, H-50), (80, H-10), (255,255,255), 2)
    cv2.putText(frame, "COLOR", (90, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

def hit_button(pt, btns):
    if pt is None:
        return None
    x, y = pt
    for name, bgr, (x1,y1,x2,y2) in btns:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return None

# ---------- Main loop ----------
W, H = 640, 480
start_cam(W, H, jpeg_quality=0.6)

paint = np.zeros((H, W, 3), dtype=np.uint8)
current_color = (0, 0, 255)  # start red paint
brush_r = 6
prev = None

btns = make_buttons(W, H)

try:
    target_fps = 8
    delay = 1.0 / target_fps

    while True:
        t0 = time.time()
        frame = grab_bgr()
        if frame is None:
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (W, H))
        frame = cv2.flip(frame, 1)  # <- add this

        c, mask = red_centroid(frame)

        # UI + overlay
        ui = frame.copy()
        draw_buttons(ui, btns, current_color)

        if c is not None:
            cx, cy = c
            # show centroid marker
            cv2.circle(ui, (cx, cy), 8, (0, 255, 255), 2)

            # If centroid hits a button -> select/clear
            hit = hit_button((cx, cy), btns)
            if hit == "R":
                current_color = (0, 0, 255)
                prev = None
            elif hit == "G":
                current_color = (0, 255, 0)
                prev = None
            elif hit == "B":
                current_color = (255, 0, 0)
                prev = None
            elif hit == "CLR":
                paint[:] = 0
                prev = None
            else:
                # Draw only if not on button area
                if prev is None:
                    cv2.circle(paint, (cx, cy), brush_r, current_color, -1)
                else:
                    cv2.line(paint, prev, (cx, cy), current_color, brush_r*2, cv2.LINE_AA)
                prev = (cx, cy)
        else:
            prev = None

        out = cv2.addWeighted(ui, 0.75, paint, 0.95, 0)
        show_in_browser(out)

        dt = time.time() - t0
        if dt < delay:
            time.sleep(delay - dt)

except KeyboardInterrupt:
    pass
finally:
    stop_cam()

