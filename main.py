import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os

VALID_MODEL = YOLO('./runs/detect/train3/weights/best.pt')

camera_active = False

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_file(file_path)

def process_file(file_path):
    results = VALID_MODEL.predict(source=file_path, show=True, save=True)
    save_dir = results[0].save_dir  # Get the directory where the results are saved
    render_results(save_dir, file_path)

def render_results(save_dir, original_file_path):
    # Check if the file is an image or video
    if original_file_path.endswith(('.png', '.jpg', '.jpeg')):
        render_results_image(save_dir)
    else:
        render_results_video(save_dir)

def render_results_image(save_dir):
    # Find the saved image in the save_dir
    for file_name in os.listdir(save_dir):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(save_dir, file_name)
            break
    else:
        print("No image found in the save directory.")
        return

    img = Image.open(img_path)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

def render_results_video(save_dir):
    for file_name in os.listdir(save_dir):
        if file_name.endswith(('.avi', '.mp4')):
            video_path = os.path.join(save_dir, file_name)
            break
    else:
        print("No video found in the save directory.")
        return

    def play_video():
        os.system(f"xdg-open {video_path}")  

    play_button = tk.Button(root, text="Play Video", command=play_video)
    play_button.pack()

def render_camera():
    global camera_active
    camera_active = True
    cap = cv2.VideoCapture(0)

    def update_frame():
        if camera_active:
            ret, frame = cap.read()
            if ret:
                results = VALID_MODEL.predict(source=frame, save=False, show=False)
                annotated_frame = results[0].plot()

                # Convert frame to RGB and display it in the Tkinter panel
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(img)
                panel.config(image=img_tk)
                panel.image = img_tk
                root.after(10, update_frame)
            else:
                stop_camera()

    def stop_camera():
        global camera_active
        camera_active = False
        cap.release()
        panel.config(image=None)

    # Add a Stop button to end the camera feed
    stop_button = tk.Button(root, text="Stop Camera", command=stop_camera)
    stop_button.pack()

    update_frame()

root = tk.Tk()
root.title("Traffic Signs Detection")

window_width = 800
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)
root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

btn_select = tk.Button(root, text="Select Image/Video", command=select_file)
btn_select.pack()

btn_camera = tk.Button(root, text="Start Camera", command=render_camera)
btn_camera.pack()

panel = tk.Label(root)
panel.pack()

# Run the GUI loop
root.mainloop()
