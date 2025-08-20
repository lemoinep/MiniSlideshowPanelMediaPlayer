# Author(s): Dr. Patrick Lemoine
# Objective: create a media panel, click on the video and play it or ...
# add cache option for movies

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
import fitz
import json
from PIL import Image  
from pillow_heif import register_heif_opener
import pillow_avif  

register_heif_opener()  # Register HEIF support

VIDEO_EXTENSIONS = ('.mp4', '.webm')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.JPG',".avif",".AVIF",".heif",".HEIF",".bmp",".BMP",".tif",".TIF")
SOUND_EXTENSIONS = ('.mp3', '.wav')
PDF_EXTENSIONS = ('.pdf', '.PDF')


def get_cache_directory(video_path):
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_name = "cache"
    cache_dir = os.path.join(video_dir, cache_name)
    return cache_dir

def save_cache_config(cache_dir, cols, rows, mode, thumb_format):
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    config_path = os.path.join(cache_dir, 'config.json')
    config_data = {'cols': cols, 'rows': rows, 'mode': mode, 'thumb_format': thumb_format}
    with open(config_path, 'w') as f:
        json.dump(config_data, f)

def load_cache_config(cache_dir):
    config_path = os.path.join(cache_dir, 'config.json')
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def has_cache_config_changed(cache_dir, cols, rows, mode, thumb_format):
    config = load_cache_config(cache_dir)
    if config is None:
        return True
    return (config.get('cols') != cols or config.get('rows') != rows or config.get('mode') != mode
            or config.get('thumb_format') != thumb_format)

def extract_frame_parallel(args):
    video_path, frame_idx = args
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return (frame_idx, None)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    return (frame_idx, img)

class Slideshow:
    def __init__(self, master, directory, panel_cols, panel_rows, mode, workers, qcache=False,thumb_format="PNG"):
        self.master = master
        self.directory = directory
        self.panel_cols = panel_cols
        self.panel_rows = panel_rows
        self.mode = mode
        self.workers = workers
        self.qcache = qcache
        self.thumb_format = thumb_format.upper()
        self.panel_step = self.panel_cols * self.panel_rows
        self.current_image = 0
        self.images = self.get_images()
        self.nb_media = len(self.images)
        self.best_grid()
        self.image_refs = []
        PathW = os.path.dirname(sys.argv[0])
        self.audio_placeholder_img = Image.open(PathW + "/audio.jpg")
        self.cache_dir = None
        if self.qcache:
            self.init_video_cache()
        self.setup_ui()
        self.update_clock()
        self.master.bind("<Escape>", self.exit_app_key)
        self.master.bind("<space>", lambda e: self.next_image())
        self.master.bind("<Left>", lambda e: self.prev_image())
        self.master.bind("<Right>", lambda e: self.next_image())



    def init_video_cache(self):
        for file_path in self.images:
            if file_path.lower().endswith(VIDEO_EXTENSIONS):
                self.cache_dir = get_cache_directory(file_path)
                break
        if self.cache_dir:
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir)
            if has_cache_config_changed(self.cache_dir, self.panel_cols, self.panel_rows, self.mode, self.thumb_format):
                # Purge old cached thumbnails
                for f in os.listdir(self.cache_dir):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.avif', '.heif')):
                        try:
                            os.remove(os.path.join(self.cache_dir, f))
                        except:
                            pass
                save_cache_config(self.cache_dir, self.panel_cols, self.panel_rows, self.mode, self.thumb_format)


    def rgb_to_hex(self, r, g, b):
        return f"#{r:02x}{g:02x}{b:02x}"

    def best_grid(self):
        if self.nb_media < self.panel_cols * self.panel_rows:
            self.panel_cols = math.ceil(math.sqrt(self.nb_media))
            self.panel_rows = math.ceil(self.nb_media / self.panel_cols)

    def shadow(self, x, y, w, h):
        colors = [(30, 30, 30, 10), (20, 20, 20, 8), (10, 10, 10, 5)]
        for r, g, b, dp in colors:
            self.canvas.create_rectangle(x+dp, y+dp, x+w+dp, y+h+dp,
                                        fill=self.rgb_to_hex(r, g, b), outline=self.rgb_to_hex(r, g, b))

    def get_images(self):
        files = [file for file in os.listdir(self.directory)
                 if file.lower().endswith(IMAGE_EXTENSIONS + VIDEO_EXTENSIONS + SOUND_EXTENSIONS + PDF_EXTENSIONS)]
        return [os.path.join(self.directory, f) for f in sorted(files)]

    def get_cached_or_generate_video_thumb(self, video_path):
        if not self.qcache or not self.cache_dir:
            return self.make_video_thumb(video_path)
        ext = self.thumb_format.lower()
        thumb_file = os.path.join(self.cache_dir, os.path.basename(video_path) + f".{ext}")
        if os.path.isfile(thumb_file):
            try:
                return Image.open(thumb_file)
            except:
                pass
        img = self.make_video_thumb(video_path)
        try:
            if self.thumb_format in ["HEIF", "AVIF"]:
                img.save(thumb_file, format=self.thumb_format, quality=90)
            elif self.thumb_format in ["JPG", "JPEG"]:
                img.save(thumb_file, format="JPEG")
            else:
                img.save(thumb_file, format="PNG")
        except:
            pass
        return img

    def make_video_thumb(self, video_path):
        if self.mode == 0:
            return self.get_first_non_black_frame(video_path, 120)
        elif self.mode == 1:
            return self.get_4_non_black_frames_composed_vers1(video_path, 1000)
        elif self.mode == 2:
            return self.get_non_black_frames_composed_vers2(video_path, 4)
        elif self.mode == 3:
            return self.get_non_black_frames_composed_vers2(video_path, 9)
        elif self.mode == 5:
            return self.get_non_black_frames_composed_parallel(video_path, 9, self.workers)
        else:
            return self.get_first_non_black_frame(video_path, 120)

    def is_frame_black(self, frame, threshold=20):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) < threshold

    def get_first_non_black_frame(self, video_path, max_frames_to_check=30):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return Image.new("RGB", (320, 240), color="black")
        frame_number = 0
        while frame_number < max_frames_to_check:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if not self.is_frame_black(frame, 50):
                cap.release()
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return Image.new("RGB", (320, 240), color="black")
        img = Image.fromarray(frame)
        w0, h0 = img.size
        frame_idx = 0
        while len(frames_collected) < num_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                break
            if not self.is_frame_black(frame):
                frames_collected.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            frame_idx += step
        cap.release()
        while len(frames_collected) < num_frames:
            frames_collected.append(frames_collected[-1].copy() if frames_collected else Image.new("RGB", (320, 240), "black"))
        size = (w0, h0)
        frames_resized = [img.resize(size) for img in frames_collected]
        grid_size = int(math.sqrt(num_frames))
        composed_img = Image.new("RGB", (size[0]*grid_size, size[1]*grid_size))
        positions = [(x*size[0], y*size[1]) for y in range(grid_size) for x in range(grid_size)]
        for pos, img in zip(positions, frames_resized):
            composed_img.paste(img, pos)
        # resize thumb    
        h1 = 540
        ratio = h1 / h0
        composed_img = composed_img.resize((int(w0 * ratio), int(h0 * ratio)), Image.LANCZOS)
        return composed_img

    def get_non_black_frames_composed_parallel(self, video_path, num_frames=4, max_workers=4):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return Image.new("RGB", (320, 240), color="black")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        step = max(total_frames / (num_frames + 1), 1)
        frame_indices = [int(step * (i + 1)) for i in range(num_frames)]
        params = [(video_path, idx) for idx in frame_indices]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = sorted(executor.map(extract_frame_parallel, params), key=lambda x: x[0])
        frames_collected = [img for idx, img in results if img is not None]
        if not frames_collected:
            frames_collected = [Image.new("RGB", (320, 240), "black") for _ in range(num_frames)]
        while len(frames_collected) < num_frames:
            frames_collected.append(frames_collected[-1].copy())
        size = frames_collected[0].size
        frames_resized = [img.resize(size) for img in frames_collected]
        grid_size = int(math.sqrt(num_frames))
        composed_img = Image.new("RGB", (size[0]*grid_size, size[1]*grid_size))
        positions = [(x*size[0], y*size[1]) for y in range(grid_size) for x in range(grid_size)]
        for pos, img in zip(positions, frames_resized):
            composed_img.paste(img, pos)
        return composed_img

    def get_4_non_black_frames_composed_vers1(self, video_path, max_frames_to_check=200):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return Image.new("RGB", (320, 240), "black")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // max_frames_to_check, 1)
        frames_collected = []
        frame_idx = 0
        ret, frame = cap.read()
        img = Image.fromarray(frame)
        w0, h0 = img.size
        read_frames = 0
        while len(frames_collected) < 4 and read_frames < max_frames_to_check:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0 and not self.is_frame_black(frame):
                frames_collected.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            frame_idx += 1
            read_frames += 1
        cap.release()
        while len(frames_collected) < 4:
            frames_collected.append(frames_collected[-1].copy() if frames_collected else Image.new("RGB", (320, 240), "black"))
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
            return f"{int(seconds//60):02d}:{int(seconds%60):02d}"
        return "??:??"

    def get_audio_length(self, file_path):
        if file_path.lower().endswith('.mp3'):
            return MP3(file_path).info.length
        elif file_path.lower().endswith('.wav'):
            return WAVE(file_path).info.length
        else:
            return 0

    def get_first_page_from_pdf(self, pdf_path, dpi=200):
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pdf_document.close()
        return img

    def get_pdf_page_count(self, pdf_path):
        pdf_document = fitz.open(pdf_path)
        page_count = pdf_document.page_count
        pdf_document.close()
        return page_count

    def get_creation_date(self, file_path):
        return datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%d/%m/%Y %H:%M")

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
        self.slider = tk.Scale(slider_frame, from_=0, to=max(len(self.images)-1, 0), orient="horizontal",
                               command=self.slider_changed, length=600, bg="#333333",
                               fg="white", troughcolor="#555555", highlightthickness=0)
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
        cols, rows = self.panel_cols, self.panel_rows
        e = 80
        max_img_width = self.screen_width // cols
        max_img_height = self.screen_height // rows - e
        start_index = self.current_image
        px = (self.screen_width - (cols * max_img_width)) // 2
        py = (self.screen_height - (rows * max_img_height)) // 2
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
                if ext in VIDEO_EXTENSIONS:
                    img = self.get_cached_or_generate_video_thumb(file_path)
                    ratio = min(Iw / img.width, Ih / img.height)
                    w, h = int(img.width * ratio), int(img.height * ratio)
                    img = img.resize((w, h), Image.LANCZOS)
                    photo_img = ImageTk.PhotoImage(img)
                    x = x + (max_img_width - w) // 2
                    y = y + (max_img_height - h) // 2
                    self.shadow(x, y, w, h)
                    img_id = self.canvas.create_image(x, y, anchor="nw", image=photo_img)
                    self.image_refs.append(photo_img)
                    self.canvas.tag_bind(img_id, "<Button-1>", lambda e, path=file_path: self.open_with_default_player(path))
                    self.canvas.create_text(x+w//2, y+h//2, text="▶", fill="white", font=("Helvetica", max(20, w//6), "bold"))
                    self.canvas.create_text(x+w//2, y+h+20, text=f"{self.get_video_duration(file_path)}  |  {self.get_creation_date(file_path)}", fill="white", font=("Helvetica", 8, "bold"))

                elif ext in SOUND_EXTENSIONS:
                    img = self.audio_placeholder_img.copy()
                    ratio = min(Iw / img.width, Ih / img.height)
                    w, h = int(img.width * ratio), int(img.height * ratio)
                    img = img.resize((w, h), Image.LANCZOS)
                    photo_img = ImageTk.PhotoImage(img)
                    x = x + (max_img_width - w) // 2
                    y = y + (max_img_height - h) // 2
                    self.shadow(x, y, w, h)
                    img_id = self.canvas.create_image(x, y, anchor="nw", image=photo_img)
                    self.image_refs.append(photo_img)
                    self.canvas.tag_bind(img_id, "<Button-1>", lambda e, path=file_path: self.open_with_default_audio_player(path))
                    self.canvas.create_text(x+w//2, y+h//2, text="▶", fill="white", font=("Helvetica", max(20, w//6), "bold"))
                    self.canvas.create_text(x+w//2, y+h+20, text=f"{self.get_audio_length(file_path):.2f} s | {self.get_creation_date(file_path)}", fill="white", font=("Helvetica", 8, "bold"))

                elif ext in PDF_EXTENSIONS:
                    img = self.get_first_page_from_pdf(file_path)
                    ratio = min(Iw / img.width, Ih / img.height)
                    w, h = int(img.width * ratio), int(img.height * ratio)
                    img = img.resize((w, h), Image.LANCZOS)
                    photo_img = ImageTk.PhotoImage(img)
                    x = x + (max_img_width - w) // 2
                    y = y + (max_img_height - h) // 2
                    self.shadow(x, y, w, h)
                    img_id = self.canvas.create_image(x, y, anchor="nw", image=photo_img)
                    self.image_refs.append(photo_img)
                    self.canvas.tag_bind(img_id, "<Button-1>", lambda e, path=file_path: self.open_with_default_player(path))
                    self.canvas.create_text(x+w//2, y+h//2, text="▶", fill="white", font=("Helvetica", max(20, w//6), "bold"))
                    self.canvas.create_text(x+w//2, y+h+20, text=f"{self.get_pdf_page_count(file_path)} Pg | {self.get_creation_date(file_path)}", fill="white", font=("Helvetica", 8, "bold"))

                else:
                    img = Image.open(file_path)
                    ratio = min(Iw / img.width, Ih / img.height)
                    w, h = int(img.width * ratio), int(img.height * ratio)
                    img = img.resize((w, h), Image.LANCZOS)
                    photo_img = ImageTk.PhotoImage(img)
                    x = x + (max_img_width - w) // 2
                    y = y + (max_img_height - h) // 2
                    self.shadow(x, y, w, h)
                    img_id = self.canvas.create_image(x, y, anchor="nw", image=photo_img)
                    self.image_refs.append(photo_img)
                    self.canvas.tag_bind(img_id, "<Button-1>", lambda e, path=file_path: self.open_with_default_image_viewer(path))
                    self.canvas.create_text(x+w//2, y+h+20, text=f"{self.get_creation_date(file_path)}", fill="white", font=("Helvetica", 8, "bold"))
        self.slider.set(self.current_image)

    def open_with_default_player(self, video_path):
        if sys.platform.startswith('darwin'):
            subprocess.Popen(['open', video_path])
        elif os.name == 'nt':
            os.startfile(video_path)
        elif os.name == 'posix':
            subprocess.Popen(['xdg-open', video_path])

    def open_with_default_image_viewer(self, image_path):
        if sys.platform.startswith('darwin'):
            subprocess.Popen(['open', image_path])
        elif os.name == 'nt':
            os.startfile(image_path)
        elif os.name == 'posix':
            subprocess.Popen(['xdg-open', image_path])

    def open_with_default_audio_player(self, audio_path):
        if sys.platform.startswith('darwin'):
            subprocess.Popen(['open', audio_path])
        elif os.name == 'nt':
            os.startfile(audio_path)
        elif os.name == 'posix':
            subprocess.Popen(['xdg-open', audio_path])

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
    def on_mouse_wheel(self, event):
        (self.prev_image() if event.delta > 0 else self.next_image())
    def on_mouse_wheel_up(self, event): self.prev_image()
    def on_mouse_wheel_down(self, event): self.next_image()

    def update_clock(self):
        self.clock_label.config(text=time.strftime('%H:%M:%S'))
        self.master.after(1000, self.update_clock)
    def exit_app_key(self, event): self.master.destroy()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--Path', type=str, default='.', help='Path.')
    parser.add_argument('--Cols', type=int, default=7, help='Columns.')
    parser.add_argument('--Rows', type=int, default=5, help='Rows.')
    parser.add_argument('--Mode', type=int, default=0, help='Mode.')
    parser.add_argument('--Workers', type=int, default=4, help='Number of threads')
    parser.add_argument('--QCache', type=int, default=0, help='Enable video thumbnail cache')
    parser.add_argument('--ThumbFormat', type=str, choices=["JPG", "PNG", "HEIF", "AVIF"], default="PNG",
                       help="Format for saved thumbnails (JPG, PNG, HEIF, AVIF)")
    args = parser.parse_args()
    root = tk.Tk()
    slideshow = Slideshow(root, args.Path, args.Cols, args.Rows, args.Mode, args.Workers, args.QCache,args.ThumbFormat)
    root.mainloop()
