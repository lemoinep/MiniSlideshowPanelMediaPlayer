# Author(s): Dr. Patrick Lemoine
# Objective: create a video panel, click on the video and play it.

import tkinter as tk
from PIL import Image, ImageTk
import os
import sys
import subprocess
import cv2
import time
import datetime

VIDEO_EXTENSIONS = ('.mp4', '.webm')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.JPG')

class Slideshow:
    def __init__(self, master, directory, panel_cols, panel_rows):
        self.master = master
        self.directory = directory
        self.panel_cols = panel_cols
        self.panel_rows = panel_rows
        self.panel_step = self.panel_cols * self.panel_rows
        self.current_image = 0
        self.images = self.get_images()
        self.image_refs = []
        self.setup_ui()
        self.update_clock()
        self.master.bind("<Escape>", self.exit_app_key)
        self.master.bind("<space>", lambda e: self.next_image())
        self.master.bind("<Left>", lambda e: self.prev_image())
        self.master.bind("<Right>", lambda e: self.next_image())

    def rgb_to_hex(self, r, g, b):
        if not all(isinstance(c, int) and 0 <= c <= 255 for c in (r, g, b)):
            raise ValueError("RGB values ​​must be whole between 0 and 255")
        return f"#{r:02x}{g:02x}{b:02x}"

    def shadow(self, x, y, w, h):
        colors = [
            (30, 30, 30, 10),
            (20, 20, 20, 8),
            (10, 10, 10, 5)
        ]
        for r, g, b, dp in colors:
            hex_color = self.rgb_to_hex(r, g, b)
            self.canvas.create_rectangle(x+dp, y+dp, x+w+dp, y+h+dp,
                                       fill=hex_color, outline=hex_color)

    def get_images(self):
        files = [file for file in os.listdir(self.directory)
                 if file.lower().endswith(IMAGE_EXTENSIONS + VIDEO_EXTENSIONS)]
        files_sorted = sorted(files)
        return [os.path.join(self.directory, file) for file in files_sorted]

    def get_video_thumbnail(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            return img
        else:
            return Image.new("RGB", (320, 240), color="black")

    def get_video_duration(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0:
            seconds = frame_count / fps
            minutes = int(seconds // 60)
            rem_sec = int(seconds % 60)
            return f"{minutes:02d}:{rem_sec:02d}"
        else:
            return "??:??"

    def get_creation_date(self, file_path):
        create_time = os.path.getctime(file_path)
        create_date = datetime.datetime.fromtimestamp(create_time)
        # Format : JJ/MM/AAAA HH:MM
        return create_date.strftime("%d/%m/%Y %H:%M")

    def setup_ui(self):
        self.master.config(bg="#333333")
        self.master.attributes('-fullscreen', True)
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()
        self.canvas = tk.Canvas(self.master, width=self.screen_width, height=self.screen_height-85,
                                bg="#333333", highlightthickness=0)
        self.canvas.pack()

        slider_frame = tk.Frame(self.master, bg="#333333")
        slider_frame.pack(fill="x", side="bottom")
        self.slider = tk.Scale(slider_frame, from_=0, to=max(len(self.images)-1, 0),
                               orient="horizontal", command=self.slider_changed,
                               length=600, bg="#333333", fg="white",
                               troughcolor="#555555", highlightthickness=0)
        self.slider.pack(padx=5, pady=5)

        self.master.bind_all("<MouseWheel>", self.on_mouse_wheel)
        self.master.bind_all("<Button-4>", self.on_mouse_wheel_up)
        self.master.bind_all("<Button-5>", self.on_mouse_wheel_down)

        self.clock_label = tk.Label(self.master, bg="#333333", fg="white", font=("Helvetica", 10))
        self.clock_label.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

        self.show_image()

    def show_image(self):
        self.canvas.delete("all")
        self.image_refs.clear()
        screen_width = self.screen_width
        screen_height = self.screen_height - 110
        cols, rows = self.panel_cols, self.panel_rows
        max_img_width = screen_width // cols - 10
        max_img_height = screen_height // rows - 50
        start_index = self.current_image
        px = (self.screen_width - cols * max_img_width) / 2         
        py = (self.screen_height - rows * max_img_height) / 2 - 70
        
                      
        for i in range(rows):
            for j in range(cols):
                if not self.images:
                    continue
                img_index = (start_index + i * cols + j) % len(self.images)
                file_path = self.images[img_index]
                ext = os.path.splitext(file_path)[1].lower()
                x = j * (screen_width // cols) + px 
                y = i * (screen_height // rows) + py
                w, h = max_img_width, max_img_height

                if ext in VIDEO_EXTENSIONS:
                    img = self.get_video_thumbnail(file_path)
                    width, height = img.size
                    ratio = min(max_img_width / width, max_img_height / height)
                    w, h = int(width * ratio), int(height * ratio)
                    img = img.resize((w, h), Image.LANCZOS)
                    photo_img = ImageTk.PhotoImage(img)
                    self.shadow(x, y, w, h)
                    img_id = self.canvas.create_image(x, y, anchor="nw", image=photo_img)
                    self.canvas.tag_bind(img_id, "<Button-1>",
                        lambda e, path=file_path: self.open_with_default_player(path))
                    self.image_refs.append(photo_img)
                    self.canvas.create_text(x+w//2, y+h//2, text="▶", fill="white",
                                           font=("Helvetica", max(20, w//6), "bold"))
                    # Addd vignette
                    duration = self.get_video_duration(file_path)
                    creation_date = self.get_creation_date(file_path)
                    info_text = f"{duration}  |  {creation_date}"
                    self.canvas.create_text(
                        x + w//2, y + h + 20,
                        text=info_text,
                        fill="white",
                        font=("Helvetica", 8, "bold")
                    )
                else:
                    img = Image.open(file_path)
                    img = img.resize((w, h), Image.LANCZOS)
                    photo_img = ImageTk.PhotoImage(img)
                    self.canvas.create_image(x, y, anchor="nw", image=photo_img)
                    self.image_refs.append(photo_img)

        self.slider.set(self.current_image)

    def open_with_default_player(self, video_path):
        if sys.platform.startswith('darwin'):
            subprocess.Popen(['open', video_path])
        elif os.name == 'nt':
            os.startfile(video_path)
        elif os.name == 'posix':
            subprocess.Popen(['xdg-open', video_path])
        else:
            print("Unsupported OS: cannot open video.")

    def next_image(self):
        if self.images:
            self.current_image = (self.current_image + self.panel_step) % len(self.images)
            self.show_image()

    def prev_image(self):
        if self.images:
            self.current_image = (self.current_image - self.panel_step) % len(self.images)
            self.show_image()

    def slider_changed(self, value):
        index = int(value)
        if 0 <= index < len(self.images):
            self.current_image = index
            self.show_image()

    def exit_fullscreen(self):
        self.master.attributes('-fullscreen', False)

    def exit_app_key(self, event):
        self.master.destroy()

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.prev_image()
        else:
            self.next_image()

    def on_mouse_wheel_up(self, event):
        self.prev_image()

    def on_mouse_wheel_down(self, event):
        self.next_image()

    def update_clock(self):
        current_time = time.strftime('%H:%M:%S')
        self.clock_label.config(text=current_time)
        self.master.after(1000, self.update_clock)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--Path', type=str, default='.', help='Path.')
    parser.add_argument('--Cols', type=int, default=7, help='Cols.')
    parser.add_argument('--Rows', type=int, default=5, help='Rows.')
    args = parser.parse_args()
    root = tk.Tk()
    slideshow = Slideshow(root, args.Path, args.Cols, args.Rows)
    root.mainloop()

