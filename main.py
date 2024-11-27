import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import time
import subprocess
import platform 
import csv

# Load the model
VALID_MODEL = YOLO('./runs/detect/train3/weights/best.pt')
DETECTED_SIGNS_FILE = "detected_signs.csv"
CLASS_MAPPING = {
    0: "Green Light",
    1: "Red Light",
    2: "Speed Limit 10",
    3: "Speed Limit 100",
    4: "Speed Limit 110",
    5: "Speed Limit 120",
    6: "Speed Limit 20",
    7: "Speed Limit 30",
    8: "Speed Limit 40",
    9: "Speed Limit 50",
    10: "Speed Limit 60",
    11: "Speed Limit 70",
    12: "Speed Limit 80",
    13: "Speed Limit 90",
    14: "Stop",
}

camera_active = False
theme = "light"


def initialize_file():
    try:
        with open(DETECTED_SIGNS_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Sign Name", "Confidence", "Bounding Box"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to initialize file: {e}")

def save_detection_to_file(sign_name, confidence, bbox):
    try:
        # Convert tensor data to Python native types
        sign_name = str(sign_name) 
        confidence = float(confidence)  
        bbox = bbox.tolist() 

        with open(DETECTED_SIGNS_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([sign_name, f"{confidence:.2f}", bbox])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save detection: {e}")


initialize_file()

def toggle_theme():
    global theme
    if theme == "light":
        root.tk_setPalette(background="#2E2E2E", foreground="#FFFFFF")
        status_bar.config(bg="#2E2E2E", fg="#FFFFFF")
        frame_top.config(bg="#2E2E2E")
        frame_buttons.config(bg="#2E2E2E")
        title.config(bg="#2E2E2E", fg="#FFFFFF")
        instructions.config(bg="#2E2E2E", fg="#FFFFFF")
        theme = "dark"
    else:
        root.tk_setPalette(background="#FFFFFF", foreground="#000000")
        status_bar.config(bg="#FFFFFF", fg="#000000")
        frame_top.config(bg="#FFFFFF")
        frame_buttons.config(bg="#FFFFFF")
        title.config(bg="#FFFFFF", fg="#000000")
        instructions.config(bg="#FFFFFF", fg="#000000")
        theme = "light"

def select_file():
    current_os = platform.system() 
    file_path = None

    try:
        if current_os == "Linux":
            # Use zenity on Linux
            result = subprocess.run(
                ["zenity", "--file-selection", "--title=Select an Image or Video"],
                capture_output=True,
                text=True
            )
            file_path = result.stdout.strip() if result.returncode == 0 else None
        elif current_os == "Darwin":  # macOS
            # Use AppleScript for macOS native file picker
            result = subprocess.run(
                ["osascript", "-e", 'POSIX path of (choose file with prompt "Select an Image or Video")'],
                capture_output=True,
                text=True
            )
            file_path = result.stdout.strip() if result.returncode == 0 else None
        else:  # Windows or fallback
            file_path = filedialog.askopenfilename(title="Select an Image or Video")

        if file_path:
            process_file(file_path)
        else:
            status_var.set("File selection canceled.")
    except Exception as e:
        messagebox.showerror("Error", f"File selection failed: {e}")
        status_var.set("Error occurred during file selection.")

def process_file(file_path):
    try:
        if file_path.endswith(('.png', '.jpg', '.jpeg')):
            render_results_image(file_path)
        else:
            render_results_video(file_path)
    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_var.set("Error occurred during processing.")

def update_progress(progress):
    progress_var.set(progress)
    root.update_idletasks()

def render_results_image(file_path):
    try:
        status_var.set("Processing image...")
        results = VALID_MODEL.predict(source=file_path, show=False)
        img = results[0].plot()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

        # Store detections
        for box in results[0].boxes:
            class_id = int(box.cls.item()) 
            sign_name = CLASS_MAPPING.get(class_id, "Unknown")
            confidence = box.conf.item()
            bbox = box.xyxy
            save_detection_to_file(sign_name, confidence, bbox)  
            log_text.insert(tk.END, f"Detected: {sign_name} with confidence {confidence:.2f}\n")

        status_var.set("Image processing complete.")
        log_text.insert(tk.END, f"Processed: {file_path}\nDetections: {results[0].boxes}\n")
    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_var.set("Error occurred during image rendering.")

def render_results_video(file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to open video file.")
            return

        status_var.set("Processing video...")
        update_progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = VALID_MODEL.predict(source=frame, save=False, show=False)
            rendered_frame = results[0].plot()
            rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rendered_frame)
            img = ImageTk.PhotoImage(img)
            panel.config(image=img)
            panel.image = img

            # Store detections
            for box in results[0].boxes:
                class_id = int(box.cls.item()) 
                sign_name = CLASS_MAPPING.get(class_id, "Unknown")
                confidence = box.conf.item()
                bbox = box.xyxy
                save_detection_to_file(sign_name, confidence, bbox)
                log_text.insert(tk.END, f"Detected: {sign_name} with confidence {confidence:.2f}\n")


            processed_frames += 1
            progress = (processed_frames / total_frames) * 100
            update_progress(progress)

            root.update_idletasks()
            root.update()
        cap.release()
        status_var.set("Video processing complete.")
        update_progress(100)
    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_var.set("Error occurred during video rendering.")

def start_camera():
    global camera_active
    camera_active = True
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera.")
        return

    status_var.set("Starting camera...")
    start_time = time.time()
    frames_processed = 0
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break
        results = VALID_MODEL.predict(source=frame, save=False, show=False)
        rendered_frame = results[0].plot()
        rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rendered_frame)
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        frames_processed += 1

        elapsed_time = time.time() - start_time
        fps = frames_processed / elapsed_time
        status_var.set(f"Camera running... FPS: {fps:.2f}")
        root.update_idletasks()
        root.update()

    cap.release()
    status_var.set("Camera stopped.")

def stop_camera():
    global camera_active
    camera_active = False
    status_var.set("Camera stopped.")

# Create the main window
root = tk.Tk()
root.title("Traffic Signs Detection")
root.geometry('1000x700')

# Top frame
frame_top = tk.Frame(root, pady=10)
frame_top.pack()

title = tk.Label(frame_top, text="Traffic Signs Detection", font=("Helvetica", 16))
title.pack()

instructions = tk.Label(frame_top, text="Select an image, video, or use the camera.", font=("Helvetica", 12))
instructions.pack()

# Buttons
frame_buttons = tk.Frame(root, pady=10)
frame_buttons.pack()

btn_select = tk.Button(frame_buttons, text="Select Image/Video", command=select_file)
btn_select.pack(side=tk.LEFT, padx=5)

btn_camera_start = tk.Button(frame_buttons, text="Start Camera", command=start_camera)
btn_camera_start.pack(side=tk.LEFT, padx=5)

btn_camera_stop = tk.Button(frame_buttons, text="Stop Camera", command=stop_camera)
btn_camera_stop.pack(side=tk.LEFT, padx=5)

btn_theme_toggle = tk.Button(frame_buttons, text="Toggle Dark Mode", command=toggle_theme)
btn_theme_toggle.pack(side=tk.LEFT, padx=5)

btn_quit = tk.Button(frame_buttons, text="Quit", command=root.quit)
btn_quit.pack(side=tk.LEFT, padx=5)

# Image/Video display
panel = tk.Label(root, bg="black")
panel.pack(fill=tk.BOTH, expand=True)


# Progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100, mode='determinate')
progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

# Log panel
log_frame = tk.Frame(root, padx=10, pady=10)
log_frame.pack(fill=tk.BOTH, expand=True)
log_text = tk.Text(log_frame, wrap=tk.WORD, height=10)
log_text.pack(fill=tk.BOTH, expand=True)
log_text.insert(tk.END, "Log initialized...\n")

# Status bar
status_var = tk.StringVar()
status_bar = tk.Label(root, textvariable=status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)
status_var.set("Ready")

# Start the application
root.mainloop()
