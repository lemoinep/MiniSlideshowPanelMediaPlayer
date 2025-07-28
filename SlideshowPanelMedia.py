# Author(s): Dr. Patrick Lemoine
# Objective: create a media panel, click on the video and play it or ...

import tkinter as tk
from PIL import Image, ImageTk
import os
import sys
import subprocess
import cv2
import time
import datetime
import math
import numpy as np
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from concurrent.futures import ProcessPoolExecutor


VIDEO_EXTENSIONS = ('.mp4', '.webm')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.JPG')
SOUND_EXTENSIONS = ('.mp3','.wav')

class Slideshow:
    def __init__(self, master, directory, panel_cols, panel_rows, mode, workers):
        self.master = master
        self.directory = directory
        self.panel_cols = panel_cols
        self.panel_rows = panel_rows
        self.mode = mode
        self.workers = workers
        self.panel_step = self.panel_cols * self.panel_rows
        self.current_image = 0
        self.images = self.get_images()
        self.nb_media = len(self.images)
        self.best_grid()
        self.image_refs = []
        currentDirectory = os.getcwd()
        PathW=os.path.dirname(sys.argv[0])
        self.audio_placeholder_img =  Image.open(PathW+"/audio.jpg")
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
    
    def best_grid(self):
        if (self.nb_media < self.panel_cols * self.panel_rows ):
            self.panel_cols = math.ceil(math.sqrt(self.nb_media))
            self.panel_rows = math.ceil(self.nb_media/self.panel_cols)
        
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
                 if file.lower().endswith(IMAGE_EXTENSIONS + VIDEO_EXTENSIONS + SOUND_EXTENSIONS)]
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

    def get_video_thumbnail_frame_number(self, video_path, frame_number):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return Image.new("RGB", (320, 240), color="black")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number) 
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            return img
        else:
            return Image.new("RGB", (320, 240), color="black")
        
        
    def is_frame_black(self, frame, threshold=20):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        return mean_intensity < threshold

    def get_first_non_black_frame(self, video_path, max_frames_to_check=30):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return Image.new("RGB", (320, 240), color="black")

        frame_number = 0
        while frame_number < max_frames_to_check:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if not self.is_frame_black(frame,50):
                cap.release()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
            frame_number += 1

        cap.release()
        return Image.new("RGB", (320, 240), color="black")
    
    
    def get_non_black_frames_composed_vers2(self, video_path, num_frames=4):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return Image.new("RGB", (320, 240), color="black")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames / (num_frames + 1), 1)
        
        frames_collected = []
        frame_idx = 0
    
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return Image.new("RGB", (320, 240), color="black")
        img = Image.fromarray(frame)
        w0, h0 = img.size
    
        while len(frames_collected) < num_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                break
            if not self.is_frame_black(frame):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                frames_collected.append(img)
            frame_idx += step
        
        cap.release()
    
        while len(frames_collected) < num_frames:
            if frames_collected:
                frames_collected.append(frames_collected[-1].copy())
            else:
                black_img = Image.new("RGB", (320, 240), color="black")
                frames_collected.extend([black_img.copy() for _ in range(num_frames)])
    
        size = (w0, h0)
        frames_resized = [img.resize(size) for img in frames_collected]
    
        import math
        grid_size = int(math.sqrt(num_frames))
        composed_img = Image.new("RGB", (size[0]*grid_size, size[1]*grid_size))
    
        positions = [(x*size[0], y*size[1]) for y in range(grid_size) for x in range(grid_size)]
    
        for pos, img in zip(positions, frames_resized):
            composed_img.paste(img, pos)
        
        return composed_img


    def get_non_black_frames_composed_parallel(self, video_path, num_frames=4, max_workers=4):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return Image.new("RGB", (320, 240), color="black")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        step = max(total_frames / (num_frames + 1), 1)
        frame_indices = [int(step * (i+1)) for i in range(num_frames)]
        
        frames_collected = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda idx: self.get_video_thumbnail_frame_number(video_path, idx), frame_indices))
        

        results.sort(key=lambda x: x[0])
        for idx, img in results:
            if img is not None and not self.is_frame_black(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)):
                frames_collected.append(img)
        
        if frames_collected:
            while len(frames_collected) < num_frames:
                frames_collected.append(frames_collected[-1].copy())
        else:
            black_img = Image.new("RGB", (320, 240), color="black")
            frames_collected = [black_img.copy() for _ in range(num_frames)]
    
        size = frames_collected[0].size
        frames_resized = [img.resize(size) for img in frames_collected]
    
        grid_size = int(math.sqrt(num_frames))
        composed_img = Image.new("RGB", (size[0]*grid_size, size[1]*grid_size))
        positions = [(x*size[0], y*size[1]) for y in range(grid_size) for x in range(grid_size)]
    
        for pos, img in zip(positions, frames_resized):
            composed_img.paste(img, pos)
        
        return composed_img


    def get_4_non_black_frames_composed_vers1(self, video_path, max_frames_to_check=200):
        # version for webm 
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return Image.new("RGB", (320, 240), color="black")
    
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // max_frames_to_check, 1)
        
        frames_collected = []
        frame_idx = 0
        read_frames = 0
        w0 = 320
        h0 = 240
        
        ret, frame = cap.read()
        img = Image.fromarray(frame)
        w0, h0 = img.size
    
        while len(frames_collected) < 4 and read_frames < max_frames_to_check:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                if not self.is_frame_black(frame):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_collected.append(Image.fromarray(frame_rgb))
            frame_idx += 1
            read_frames += 1
    
        cap.release()
    
        if len(frames_collected) == 0:
            black_img = Image.new("RGB", (320, 240), color="black")
            frames_collected = [black_img.copy() for _ in range(4)]
        else:
            while len(frames_collected) < 4:
                frames_collected.append(frames_collected[-1].copy())
    
        size = (w0, h0)        
        frames_resized = [img.resize(size) for img in frames_collected]    
        composed_img = Image.new("RGB", (size[0]*2, size[1]*2))
        positions = [(0,0), (size[0],0), (0,size[1]), (size[0],size[1])]
        for pos, img in zip(positions, frames_resized):
            composed_img.paste(img, pos)
    
        return composed_img
        
    
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
        
    def get_audio_length(self,file_path):
        if file_path.lower().endswith('.mp3'):
            audio = MP3(file_path)
            return audio.info.length
        elif file_path.lower().endswith('.wav'):
            audio = WAVE(file_path)
            return audio.info.length
        else:
            raise ValueError("Error not valid")

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
        screen_height = self.screen_height
        cols, rows = self.panel_cols, self.panel_rows
        e = 80       
        max_img_width = screen_width // cols
        max_img_height = screen_height // rows - e
        start_index = self.current_image

        grid_width = cols * max_img_width
        grid_height = rows * max_img_height
        
        px = (screen_width - grid_width) // 2 
        py = (screen_height - grid_height) // 2
        
        Iw = max_img_width - e
        Ih = max_img_height - e

        for i in range(rows):
            for j in range(cols):
                if not self.images:
                    continue
                img_index = (start_index + i * cols + j) % len(self.images)
                file_path = self.images[img_index]
                ext = os.path.splitext(file_path)[1].lower()

                x = px + j * (max_img_width)
                y = py + i * (max_img_height)
                w, h = max_img_width, max_img_height

                if ext in VIDEO_EXTENSIONS:
                    #img = self.get_video_thumbnail(file_path)
                    
                    if (self.mode==0) :
                        img = self.get_first_non_black_frame(file_path,120) 
                    
                    if (self.mode==1) :
                        img = self.get_4_non_black_frames_composed_vers1(file_path,1000)
                        
                    if (self.mode==2) :
                        img = self.get_non_black_frames_composed_vers2(file_path, 4)
                        
                    if (self.mode==3) :
                        img = self.get_non_black_frames_composed_vers2(file_path, 9)
                        
                    if (self.mode==5) :
                        img = self.get_non_black_frames_composed_parallel(file_path, 9, self.workers)   

                    
                    width, height = img.size
                    ratio = min(Iw / width, Ih / height)
                    w, h = int(width * ratio), int(height * ratio)
                    img = img.resize((w, h), Image.LANCZOS)
                    photo_img = ImageTk.PhotoImage(img)
                    x = x + (max_img_width - w) // 2  
                    y = y + (max_img_height - h) // 2  
                    self.shadow(x, y, w, h)
                    img_id = self.canvas.create_image(x, y, anchor="nw", image=photo_img)
                    self.canvas.tag_bind(img_id, "<Button-1>",
                        lambda e, path=file_path: self.open_with_default_player(path))
                    self.image_refs.append(photo_img)
                    self.canvas.create_text(x+w//2, y+h//2, text="▶", fill="white",
                                           font=("Helvetica", max(20, w//6), "bold"))
                    duration = self.get_video_duration(file_path)
                    creation_date = self.get_creation_date(file_path)
                    info_text = f"{duration}  |  {creation_date}"
                    self.canvas.create_text(
                        x + w//2, y + h + 20,
                        text=info_text,
                        fill="white",
                        font=("Helvetica", 8, "bold")
                    )
                    
                elif ext in SOUND_EXTENSIONS:
                    img = self.audio_placeholder_img.copy()
                    width, height = img.size
                    ratio = min(Iw / width, Ih / height)
                    w, h = int(width * ratio), int(height * ratio)
                    img = img.resize((w, h), Image.LANCZOS)
                    photo_img = ImageTk.PhotoImage(img)
                    x = x + (max_img_width - w) // 2  
                    y = y + (max_img_height - h) // 2  
                    self.shadow(x, y, w, h)
                    img_id = self.canvas.create_image(x, y, anchor="nw", image=photo_img)
                    self.canvas.tag_bind(img_id, "<Button-1>",
                        lambda e, path=file_path: self.open_with_default_player(path))
                    self.image_refs.append(photo_img)
                    self.canvas.create_text(x+w//2, y+h//2,text="▶",fill="white",
                                            font=("Helvetica", max(20, w//6), "bold"))
                      
                    duration = self.get_audio_length(file_path)                      
                    creation_date = self.get_creation_date(file_path)
                    info_text = f"{duration:.2f} s |  {creation_date}"
                    self.canvas.create_text(
                        x + w//2, y + h + 20,
                        text=info_text,
                        fill="white",
                        font=("Helvetica", 8, "bold")
                    )
                else:
                    img = Image.open(file_path)
                    width, height = img.size
                    ratio = min(Iw / width, Ih / height)
                    w, h = int(width * ratio), int(height * ratio)
                    img = img.resize((w, h), Image.LANCZOS)
                    photo_img = ImageTk.PhotoImage(img)
                    x = x + (max_img_width - w) // 2  
                    y = y + (max_img_height - h) // 2  
                    self.shadow(x, y, w, h)
                    img_id = self.canvas.create_image(x, y, anchor="nw", image=photo_img)
                    self.canvas.tag_bind(img_id, "<Button-1>",
                        lambda e, path=file_path: self.open_with_default_image_viewer(path))
                    self.image_refs.append(photo_img)
                    
                    creation_date = self.get_creation_date(file_path)
                    info_text = f"{creation_date}"
                    self.canvas.create_text(
                        x + w//2, y + h + 20,
                        text=info_text,
                        fill="white",
                        font=("Helvetica", 8, "bold")
                    )

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

    def open_with_default_image_viewer(self, image_path):
        if sys.platform.startswith('darwin'):
            subprocess.Popen(['open', image_path])
        elif os.name == 'nt':
            os.startfile(image_path)
        elif os.name == 'posix':
            subprocess.Popen(['xdg-open', image_path])
        else:
            print("Unsupported OS: cannot open image.")
            
    def open_with_default_audio_player(self, audio_path):
        if sys.platform.startswith('darwin'):
            subprocess.Popen(['open', audio_path])
        elif os.name == 'nt':
            os.startfile(audio_path)
        elif os.name == 'posix':
            subprocess.Popen(['xdg-open', audio_path])
        else:
            print("Unsupported OS: cannot open audio file.")

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
    parser.add_argument('--Mode', type=int, default=0, help='Mode.')
    parser.add_argument('--Workers', type=int, default=4, help='Number of parallel threads')
    args = parser.parse_args()
    root = tk.Tk()
    slideshow = Slideshow(root, args.Path, args.Cols, args.Rows, args.Mode, args.Workers)
    root.mainloop()

