import cv2
import pygame
import numpy as np
import tkinter as tk

# -----------------------------
# Login GUI
# -----------------------------
plate_number = None

def driver_login_gui():
    def submit():
        global plate_number
        plate_number = entry.get()
        if plate_number.strip():
            window.destroy()

    window = tk.Tk()
    window.title("Driver Login")
    window.geometry("300x150")
    window.resizable(False, False)

    tk.Label(window, text="Enter Car Plate Number", font=("Arial", 12)).pack(pady=10)
    entry = tk.Entry(window, font=("Arial", 12), justify='center')
    entry.pack()
    entry.focus()

    tk.Button(window, text="Start Detection", font=("Arial", 11), command=submit).pack(pady=10)

    window.mainloop()

driver_login_gui()
print("✅ Logged in with plate number:", plate_number)

# -----------------------------
# Audio setup
# -----------------------------
pygame.mixer.init()
pygame.mixer.music.load("music.wav")  # change if needed

# -----------------------------
# Haar Cascades
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")  # local file

if mouth_cascade.empty():
    print("❌ Failed to load mouth cascade! Ensure 'haarcascade_mcs_mouth.xml' is in this folder.")
    raise SystemExit

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Webcam not accessible.")
    raise SystemExit

# -----------------------------
# Thresholds & counters
# -----------------------------
eye_closed_frames = 0
yawn_frames = 0

ALARM_THRESHOLD = 20   # frames with no detected eyes
YAWN_THRESHOLD = 5     # LOWERED threshold for faster yawn detection

alarm_active = False
status_text = "Awake"

# Night mode toggle
NIGHT_MODE = True   # start in night mode; press 'n' to toggle

# Colors (BGR)
CLR_AWAKE  = (0, 255, 0)
CLR_DROWSY = (0, 0, 255)
CLR_YAWN   = (0, 165, 255)  # orange
CLR_TEXT   = (255, 255, 255)
CLR_BAR_BG = (60, 60, 60)
CLR_BAR_FILL = (0, 200, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX

def trigger_alarm():
    global alarm_active
    if not alarm_active and not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()
    alarm_active = True

def stop_alarm():
    global alarm_active
    if alarm_active:
        pygame.mixer.music.stop()
    alarm_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_detected = 0
    yawn_detected = False

    for (x, y, w, h) in faces:
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=3)
        eyes_detected += len(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 1)

        half_h = h // 2
        mouth_roi_gray = face_roi_gray[half_h:, :]
        mouths = mouth_cascade.detectMultiScale(mouth_roi_gray, scaleFactor=1.5, minNeighbors=11)

        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(
                face_roi_color,
                (mx, half_h + my),
                (mx + mw, half_h + my + mh),
                (0, 0, 255),
                1
            )
            yawn_detected = True
            break

    if eyes_detected == 0:
        eye_closed_frames += 1
    else:
        eye_closed_frames = 0

    if yawn_detected:
        yawn_frames += 1
    else:
        yawn_frames = 0

    if yawn_frames > YAWN_THRESHOLD:
        status_text = "Yawning"
        trigger_alarm()
    elif eye_closed_frames > ALARM_THRESHOLD:
        status_text = "Drowsy"
        trigger_alarm()
    else:
        status_text = "Awake"
        stop_alarm()

    if NIGHT_MODE:
        overlay = frame.copy()
        dark = np.zeros_like(frame)
        alpha = 0.5
        cv2.addWeighted(overlay, 1 - alpha, dark, alpha, 0, frame)

    if status_text == "Awake":
        status_color = CLR_AWAKE
    elif status_text == "Drowsy":
        status_color = CLR_DROWSY
    else:
        status_color = CLR_YAWN

    # On-screen status
    cv2.putText(frame, f"Plate: {plate_number}", (10, 20), FONT, 0.6, CLR_TEXT, 1)
    cv2.putText(frame, f"Status: {status_text}", (10, 45), FONT, 0.7, status_color, 2)
    cv2.putText(frame, f"Eye Closed Frames: {eye_closed_frames}", (10, 70), FONT, 0.6, CLR_TEXT, 1)
    cv2.putText(frame, f"Yawn Frames: {yawn_frames}", (10, 95), FONT, 0.6, CLR_TEXT, 1)

    bar_w = 200
    bar_h = 12
    base_y = 120

    pct_eye = min(eye_closed_frames / ALARM_THRESHOLD, 1.0)
    cv2.rectangle(frame, (10, base_y), (10 + bar_w, base_y + bar_h), CLR_BAR_BG, -1)
    cv2.rectangle(frame, (10, base_y), (10 + int(bar_w * pct_eye), base_y + bar_h), CLR_BAR_FILL, -1)
    cv2.putText(frame, "Drowsy Threshold", (10 + bar_w + 10, base_y + bar_h - 2), FONT, 0.5, CLR_TEXT, 1)

    base_y2 = base_y + 20
    pct_yawn = min(yawn_frames / YAWN_THRESHOLD, 1.0)
    cv2.rectangle(frame, (10, base_y2), (10 + bar_w, base_y2 + bar_h), CLR_BAR_BG, -1)
    cv2.rectangle(frame, (10, base_y2), (10 + int(bar_w * pct_yawn), base_y2 + bar_h), CLR_BAR_FILL, -1)
    cv2.putText(frame, "Yawn Threshold", (10 + bar_w + 10, base_y2 + bar_h - 2), FONT, 0.5, CLR_TEXT, 1)

    cv2.putText(frame, "[n] Night Mode  [q] Quit", (10, frame.shape[0] - 10), FONT, 0.5, CLR_TEXT, 1)

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        NIGHT_MODE = not NIGHT_MODE
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
pygame.mixer.quit()
