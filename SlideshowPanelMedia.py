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
from pathlib import Path 

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


def cv_save_image_to_avif(img, output_path, quality=80):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.save(output_path, 'AVIF', quality=quality)

def cv_load_image_avif(path):
    pil_img = Image.open(path).convert("RGB")
    img_np = np.array(pil_img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_cv

def CV_Sharpen2d(source, alpha, gamma, num_op):
    def sharpen_kernel(src):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(src, -1, kernel)

    dst = sharpen_kernel(source)

    if num_op == 1:
        source_filtered = cv2.GaussianBlur(source, (3, 3), 0)
    elif num_op == 2:
        source_filtered = cv2.blur(source, (9, 9))
    else:
        source_filtered = source.copy()

    dst_img = cv2.addWeighted(source_filtered, alpha, dst, 1.0 - alpha, gamma)
    return dst_img

def CV_EnhanceColor(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255) 
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.1, 0, 255)  
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def CV_Vibrance2D(img, saturation_scale=1.3, brightness_scale=1.1, apply=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def CV_AdjustBrightnessContrast(img,brightness=10,contrast=2.3): 
    imgR = cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness) 
    return imgR


def CV_AdaptativeContrast(img,clip=9):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    clahe=cv2.createCLAHE(clipLimit=clip,tileGridSize=(8,8))
    merged=cv2.merge((clahe.apply(l),a,b))
    dest=cv2.cvtColor(merged,cv2.COLOR_LAB2BGR)
    return (dest)


def CV_CLAHE(img, clipLimit=2.0, tileGridSize=(8,8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def CV_SaliencyAddWeighted(img, alpha=0.6, beta=0.4, gamma=0):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliencyMap = saliency.computeSaliency(img)
    if not success:
        return img
    saliencyMap = (saliencyMap * 255).astype(np.uint8)
    saliencyMap_color = cv2.cvtColor(saliencyMap, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, alpha, saliencyMap_color, beta, gamma)

def CV_Grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def CV_Entropy(img, block_size=(32, 32)):
    gray_img= CV_Grayscale(img)
    h, w = gray_img.shape
    bh, bw = block_size
    n_blocks_y = h // bh
    n_blocks_x = w // bw
    entrop_map = np.zeros((n_blocks_y, n_blocks_x), dtype=np.float32)

    for y in range(n_blocks_y):
        for x in range(n_blocks_x):
            block = gray_img[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
            hist, _ = np.histogram(block, bins=256, range=(0,256), density=True)
            hist = hist + 1e-7  # pour éviter log(0)
            entrop_map[y, x] = -np.sum(hist * np.log2(hist))

    entrop_map_norm = cv2.normalize(entrop_map, None, 0, 255, cv2.NORM_MINMAX)
    entrop_map_norm = entrop_map_norm.astype(np.uint8)
    entrop_img = cv2.resize(entrop_map_norm, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return entrop_img

def CV_Stereo_Anaglyph(img_stereo, parallax_offset=0, lim_ratio=2.5):
    height, width, _ = img_stereo.shape

    ratio = width / height
    if ratio > lim_ratio:
        img_left = img_stereo[:, :width//2, :]
        img_right = img_stereo[:, width//2:, :]
    else:
        img_left = img_stereo
        img_right = img_stereo

    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    if parallax_offset != 0:
        M = np.float32([[1, 0, parallax_offset], [0, 1, 0]])
        img_right_gray = cv2.warpAffine(
            img_right_gray,
            M,
            (img_right_gray.shape[1], img_right_gray.shape[0]),
            borderMode=cv2.BORDER_REPLICATE
        )

    min_height = min(img_left_gray.shape[0], img_right_gray.shape[0])
    min_width = min(img_left_gray.shape[1], img_right_gray.shape[1])
    img_left_gray = img_left_gray[:min_height, :min_width]
    img_right_gray = img_right_gray[:min_height, :min_width]

    anaglyph = np.zeros((min_height, min_width, 3), dtype=img_left.dtype)
    anaglyph[..., 0] = img_right_gray  # Blue
    anaglyph[..., 1] = img_right_gray  # Green
    anaglyph[..., 2] = img_left_gray   # Red
    
    return anaglyph


def view_picture_zoom(image_path):
 
    if image_path.lower().endswith(('.avif','.heif')):
        img = cv_load_image_avif(image_path)
    else:    
        img = cv2.imread(image_path)
              
    zoom_scale = 1.0
    zoom_min = 1.0
    zoom_max = 15.0
    mouse_x, mouse_y = -1, -1
    height, width = img.shape[:2]
    qLoop = True
    qSharpen = False
    qEnhanceColor = False
    qVibrance = False
    qSaliency = False
    qClache   = False
    qBrightnessContrast = False
    qAdaptativeContrast = False
    qEntropy = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal zoom_scale, mouse_x, mouse_y, qLoop 
        mouse_x, mouse_y = x, y
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags < 0:
                zoom_scale = min(zoom_scale + 0.1, zoom_max)
            else:
                zoom_scale = max(zoom_scale - 0.1, zoom_min)
                
        if event == cv2.EVENT_RBUTTONDOWN:
            qLoop = False

    def get_zoomed_image(image, scale, center_x, center_y):
        h, w = image.shape[:2]
        new_w = int(w / scale)
        new_h = int(h / scale)

        left = max(center_x - new_w // 2, 0)
        right = min(center_x + new_w // 2, w)
        top = max(center_y - new_h // 2, 0)
        bottom = min(center_y + new_h // 2, h)

        if right - left < new_w:
            if left == 0:
                right = new_w
            elif right == w:
                left = w - new_w
        if bottom - top < new_h:
            if top == 0:
                bottom = new_h
            elif bottom == h:
                top = h - new_h

        cropped = image[top:bottom, left:right]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return zoomed

    window_name = 'Picture Zoom'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    ratio = width / height
    lh = 900
    lw = int(lh * ratio)
    cv2.resizeWindow(window_name, lw, lh)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    screen_width = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) or 1920  
    screen_height = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) or 1080  

    start_x = int((screen_width - lw) / 2)
    start_y = int((screen_height - lh) / 2)
    cv2.moveWindow(window_name, start_x, start_y)

    while qLoop:
        if mouse_x == -1 and mouse_y == -1:
            mouse_x, mouse_y = width // 2, height // 2

        zoomed_img = get_zoomed_image(img, zoom_scale, mouse_x, mouse_y)
        if qClache  :
            zoomed_img = CV_CLAHE(zoomed_img)
        if qSharpen :
            zoomed_img = CV_Sharpen2d(zoomed_img, 0.1, 0.0,  1)       
        if qEnhanceColor :
            zoomed_img = CV_EnhanceColor(zoomed_img)
        if qVibrance :
            zoomed_img = CV_Vibrance2D(zoomed_img)
        if qBrightnessContrast :
            zoomed_img = CV_AdjustBrightnessContrast(zoomed_img)
        if qAdaptativeContrast :
            zoomed_img = CV_AdaptativeContrast(zoomed_img)
        if qSaliency :
            zoomed_img = CV_SaliencyAddWeighted(zoomed_img)
        if qEntropy:
            zoomed_img = CV_Entropy(zoomed_img)
            
        
        
        cv2.imshow(window_name, zoomed_img)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            path = Path(image_path).parent
            new_path = path / "Screenshot"
            new_path .mkdir(parents=True, exist_ok=True)
            date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            outputName = "Noname"
            outputName = f"{outputName}_{date_time}.jpg"
            outputName = Path(new_path) / outputName
            cv2.imwrite(outputName, zoomed_img)
        elif key == ord('x'): qSharpen = not qSharpen
        elif key == ord('e'): qEnhanceColor = not qEnhanceColor
        elif key == ord('v'): qVibrance = not qVibrance
        elif key == ord('a'): qSaliency = not qSaliency
        elif key == ord('h'): qClache = not qClache
        elif key == ord('b'): qBrightnessContrast = not qBrightnessContrast
        elif key == ord('c'): qAdaptativeContrast = not qAdaptativeContrast
        elif key == ord('t'): qEntropy = not qEntropy
        elif key == ord('.'):  
            zoom_scale = 1.0
    cv2.destroyAllWindows()
    

def play_video_with_seek_and_pause(video_path):
    zoom_scale = 1.0
    zoom_min = 1.0
    zoom_max = 15.0
    mouse_x, mouse_y = -1, -1
    qLoop = True
    qSharpen = False
    qEnhanceColor = False
    qVibrance = False
    qLoopVideo = False
    qDrawLineOnImage = True
    qSaliency = False
    qClache   = False
    qBrightnessContrast = False
    qAdaptativeContrast = False
    qEntropy = False
    
    def draw_line_on_image(num_frame, nb_frames,img):
        height, width = img.shape[:2]
        ratio = num_frame / nb_frames
        x = int(ratio * width)    
        image_with_line = img.copy()
        cv2.line(image_with_line,(0, height-9), (x, height-9),(0, 0, 0), 6)
        cv2.line(image_with_line,(0, height-10), (x, height-10),(0, 0, 125), 6)
        cv2.line(image_with_line,(0, height-10), (x, height-10),(0, 0, 255), 3)
        return image_with_line
    
    def mouse_click_inside(x, y, x1, y1, x2, y2):
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal zoom_scale, mouse_x, mouse_y, current_frame, paused, qLoop
        mouse_x, mouse_y = x, y
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags < 0:
                zoom_scale = min(zoom_scale + 0.1, zoom_max)
            else:
                zoom_scale = max(zoom_scale - 0.1, zoom_min)
                
        elif event == cv2.EVENT_LBUTTONDOWN:
            #clicked_frame = int((x / width) * frame_count)
            #clicked_frame = max(0, min(clicked_frame, frame_count - 1))
            #current_frame = clicked_frame
            #cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            #paused = False
            
            if mouse_click_inside(mouse_x, mouse_y, 0, 0,  width, 50):
                paused = not paused 
            else :
                clicked_frame = int((x / width) * frame_count)
                clicked_frame = max(0, min(clicked_frame, frame_count - 1))
                current_frame = clicked_frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                paused = False
            
  
        if event == cv2.EVENT_RBUTTONDOWN:
            qLoop = False
    
    def get_zoomed_image(image, scale, center_x, center_y):
        h, w = image.shape[:2]
        new_w = int(w / scale)
        new_h = int(h / scale)

        left = max(center_x - new_w // 2, 0)
        right = min(center_x + new_w // 2, w)
        top = max(center_y - new_h // 2, 0)
        bottom = min(center_y + new_h // 2, h)

        if right - left < new_w:
            if left == 0:
                right = new_w
            elif right == w:
                left = w - new_w
        if bottom - top < new_h:
            if top == 0:
                bottom = new_h
            elif bottom == h:
                top = h - new_h

        cropped = image[top:bottom, left:right]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return zoomed  
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error Open Movie.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 

    fps_movie = fps
    paused = False
    current_frame = 0

    window_name = 'Movie Player'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    ratio = width / height
    lh = 900
    lw = int(lh * ratio)
    cv2.resizeWindow(window_name, lw, lh)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    screen_width = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) or 1920  
    screen_height = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) or 1080  

    start_x = int((screen_width - lw) / 2)
    start_y = int((screen_height - lh) / 2)
    cv2.moveWindow(window_name, start_x, start_y)
    
    
    cv2.setMouseCallback(window_name, mouse_callback)

    while qLoop:
        if not paused:
            if qLoopVideo and (current_frame>=frame_count-1) :
                current_frame = 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if mouse_x == -1 or mouse_y == -1:
                mouse_x, mouse_y = width // 2, height // 2
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

        zoomed_img = get_zoomed_image(frame, zoom_scale, mouse_x, mouse_y)
        if qClache  :
            zoomed_img = CV_CLAHE(zoomed_img)
        if qSharpen :
            zoomed_img = CV_Sharpen2d(zoomed_img, 0.1, 0.0,  1)
        if qEnhanceColor :
            zoomed_img = CV_EnhanceColor(zoomed_img)
        if qVibrance :
            zoomed_img = CV_Vibrance2D(zoomed_img)
        if qBrightnessContrast :
            zoomed_img = CV_AdjustBrightnessContrast(zoomed_img)
        if qAdaptativeContrast :
            zoomed_img = CV_AdaptativeContrast(zoomed_img)
        if qSaliency :
            zoomed_img = CV_SaliencyAddWeighted(zoomed_img)
            
        if qEntropy:
            zoomed_img = CV_Entropy(zoomed_img)
            
        if qDrawLineOnImage :
            zoomed_img=draw_line_on_image(current_frame, frame_count, zoomed_img)
            
            
        cv2.imshow('Movie Player', zoomed_img)
        
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == 27:  
            break
        elif key == ord(' '): paused = not paused
        elif key == ord('2'): fps = fps_movie
        elif key == ord('1'): fps = max ( 1, fps // 2)
        elif key == ord('2'): fps = fps_movie
        elif key == ord('3'): fps = fps * 2
        elif key == ord('+'): current_frame = current_frame + 1
        elif key == ord('-'): current_frame = current_frame - 1
        elif key == ord('s'):
            path = Path(video_path).parent
            new_path = path / "Screenshot"
            new_path .mkdir(parents=True, exist_ok=True)
            date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            outputName = "Noname"
            outputName = f"{outputName}_{date_time}.jpg"
            outputName = Path(new_path) / outputName
            cv2.imwrite(outputName, zoomed_img)
        elif key == ord('x'): qSharpen = not qSharpen
        elif key == ord('e'): qEnhanceColor = not qEnhanceColor
        elif key == ord('v'): qVibrance = not qVibrance    
        elif key == ord('l'): qLoopVideo = not qLoopVideo 
        elif key == ord('L'): qDrawLineOnImage = not qDrawLineOnImage 
        elif key == ord('a'): qSaliency = not qSaliency
        elif key == ord('h'): qClache = not qClache
        elif key == ord('b'): qBrightnessContrast = not qBrightnessContrast
        elif key == ord('c'): qAdaptativeContrast = not qAdaptativeContrast
        elif key == ord('t'): qEntropy = not qEntropy
        elif key == ord('.'): zoom_scale = 1.0
            
    cap.release()
    cv2.destroyAllWindows()


class Slideshow:
    def __init__(self, master, directory, panel_cols, panel_rows, mode, workers, qcache=False,thumb_format="PNG",SortFiles="NAME", qModeSoftwareView=True):
        self.master = master
        self.directory = directory
        self.panel_cols = panel_cols
        self.panel_rows = panel_rows
        self.mode = mode
        self.workers = workers
        self.qcache = qcache
        self.qModeSoftwareView = qModeSoftwareView
        self.thumb_format = thumb_format.upper()
        self.panel_step = self.panel_cols * self.panel_rows
        self.current_image = 0
        self.images = self.get_images(SortFiles)
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
        
        self.master.bind("m", self.mode_soft_app_key)



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
    
    def get_images(self, sort_by='NAME'):
        files = [file for file in os.listdir(self.directory)
                 if file.lower().endswith(IMAGE_EXTENSIONS + VIDEO_EXTENSIONS + SOUND_EXTENSIONS + PDF_EXTENSIONS)]
        if sort_by == 'NAME':
            sorted_files = sorted(files)
        elif sort_by == 'DATE':
            sorted_files = sorted(files, key=lambda x: os.path.getctime(os.path.join(self.directory, x)))
        else:
            sorted_files = files  
    
        return [os.path.join(self.directory, f) for f in sorted_files]
    

    def get_cached_or_generate_video_thumb(self, video_path):
        if not self.qcache or not self.cache_dir:
            return self.make_video_thumb(video_path)
        ext = self.thumb_format.lower()
        thumb_file = os.path.join(self.cache_dir, os.path.basename(video_path) + f".{ext}")
        if not (os.path.isfile(thumb_file)):
            img = self.make_video_thumb(video_path)
            
        if os.path.isfile(thumb_file):
            try:
                return Image.open(thumb_file)
            except:
                pass
        #img = self.make_video_thumb(video_path)
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
                    self.canvas.tag_bind(img_id, "<Button-1>", lambda e, path=file_path: self.open_with_default_pdf_player(path))
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
        if self.qModeSoftwareView:
            if sys.platform.startswith('darwin'):
                subprocess.Popen(['open', video_path])
            elif os.name == 'nt':
                os.startfile(video_path)
            elif os.name == 'posix':
                subprocess.Popen(['xdg-open', video_path])
        else :
            play_video_with_seek_and_pause(video_path)
            
    def open_with_default_pdf_player(self, pdf_path):
        #if self.qModeSoftwareView:
        if True:
            if sys.platform.startswith('darwin'):
                subprocess.Popen(['open', pdf_path])
            elif os.name == 'nt':
                os.startfile(pdf_path)
            elif os.name == 'posix':
                subprocess.Popen(['xdg-open', pdf_path])
        else :
            play_video_with_seek_and_pause(pdf_path)

    def open_with_default_image_viewer(self, image_path):
        if self.qModeSoftwareView:
            if sys.platform.startswith('darwin'):
                subprocess.Popen(['open', image_path])
            elif os.name == 'nt':
                os.startfile(image_path)
            elif os.name == 'posix':
                subprocess.Popen(['xdg-open', image_path])        
        else :
            view_picture_zoom(image_path)

        
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
    
    def mode_soft_app_key(self, event): 
        self.qModeSoftwareView = not self.qModeSoftwareView   

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
    parser.add_argument('--QModeSoftwareView', type=int, default=1, help='Enable ModeSoftwareView')
    
    parser.add_argument('--ThumbFormat', type=str, choices=["JPG", "PNG", "HEIF", "AVIF"], default="PNG",
                       help="Format for saved thumbnails (JPG, PNG, HEIF, AVIF)")
    parser.add_argument('--SortFiles', type=str, choices=["NAME", "DATE"], default="NAME",
                       help="Sort by NAME or DATE")
    args = parser.parse_args()
    root = tk.Tk()
    slideshow = Slideshow(root, args.Path, args.Cols, args.Rows, args.Mode, args.Workers, args.QCache,args.ThumbFormat,args.SortFiles,args.QModeSoftwareView)
    root.mainloop()
