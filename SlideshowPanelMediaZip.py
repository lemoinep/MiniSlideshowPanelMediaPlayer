# Author(s): Dr. Patrick Lemoine
# Objective: create a media panel, click on the video and play it or ...
# add cache option for movies and pictures in Zip

import tkinter as tk
from tkinter import scrolledtext
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import fitz
import json
from PIL import Image  
from pillow_heif import register_heif_opener
import pillow_avif 
from pathlib import Path 
from zipfile import ZipFile, ZIP_DEFLATED, is_zipfile
from io import BytesIO
from tqdm import tqdm
import shutil
import zipfile

import pygame
from pydub import AudioSegment

from scipy.signal import spectrogram

from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from multiprocessing import Process

register_heif_opener()  # Register HEIF support

VIDEO_EXTENSIONS = ('.mp4', '.webm', '.avi','.mkv')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.JPG',".avif",".AVIF",".heif",".HEIF",".bmp",".BMP",".tif",".TIF")
SOUND_EXTENSIONS = ('.mp3', '.wav')
PDF_EXTENSIONS = ('.pdf', '.PDF')
TXT_EXTENSIONS = ('.txt', '.TXT')
MD_EXTENSIONS = ('.md', '.MD')



def get_cache_zip_path(video_path):
    video_dir = os.path.dirname(video_path)
    return os.path.join(video_dir, "cache_thumbs.zip")

def save_cache_config_to_zip(zip_path, cols, rows, mode, thumb_format):
    config_data = {'cols': cols, 'rows': rows, 'mode': mode, 'thumb_format': thumb_format}
    with ZipFile(zip_path, 'a', compression=ZIP_DEFLATED) as zipf:
        zipf.writestr('config.json', json.dumps(config_data))

def load_cache_config_from_zip(zip_path):
    if not os.path.isfile(zip_path) or not is_zipfile(zip_path):
        return None
    with ZipFile(zip_path, 'r') as zipf:
        if 'config.json' not in zipf.namelist():
            return None
        return json.loads(zipf.read('config.json').decode())

def has_cache_config_changed_zip(zip_path, cols, rows, mode, thumb_format):
    config = load_cache_config_from_zip(zip_path)
    if config is None:
        return True
    return (config.get('cols') != cols or config.get('rows') != rows or
        config.get('mode') != mode or config.get('thumb_format') != thumb_format)

def save_image_to_zip(zip_path, name, image, fmt="PNG"):
    temp_images = {}
    if os.path.isfile(zip_path) and is_zipfile(zip_path):
        with ZipFile(zip_path, 'r') as zipf:
            for f in zipf.namelist():
                if f != name:
                    temp_images[f] = zipf.read(f)
    with ZipFile(zip_path, 'w', compression=ZIP_DEFLATED) as zipf:
        for f, data in temp_images.items():
            zipf.writestr(f, data)
        bio = BytesIO()
        image.save(bio, fmt)
        zipf.writestr(name, bio.getvalue())

def load_image_from_zip(zip_path, name):
    if not os.path.isfile(zip_path) or not is_zipfile(zip_path):
        return None
    with ZipFile(zip_path, 'r') as zipf:
        if name not in zipf.namelist():
            return None
        data = zipf.read(name)
        return Image.open(BytesIO(data))

def purge_zip_except_config(zip_path):

    if not os.path.isfile(zip_path) or not is_zipfile(zip_path):
        return
    with ZipFile(zip_path, 'r') as zin:
        files = zin.namelist()
        config = zin.read('config.json') if 'config.json' in files else None
    with ZipFile(zip_path, 'w', compression=ZIP_DEFLATED) as zout:
        if config: zout.writestr('config.json', config)


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


def CV_SaliencyAddWeighted(img, alpha=0.6, beta=0.4, gamma=0, num_palette=1):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliencyMap = saliency.computeSaliency(img)
    if not success:
        return img
    saliencyMap = (saliencyMap * 255).astype(np.uint8)
    if num_palette==1 :
        saliencyMap_color = cv2.applyColorMap(saliencyMap, cv2.COLORMAP_HOT)
    else :
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

def CV_Dilate(img, d=3):
    kernel = np.ones((d,d),np.uint8)
    return cv2.dilate(img, kernel, iterations = 1)
    
def CV_Erode(img, d=3):
    kernel = np.ones((d,d),np.uint8)
    return cv2.erode(img, kernel, iterations = 1)

def CV_RemoveNoise(img):
    return cv2.medianBlur(img,5)

def CV_Opening(img, d=3):
    kernel = np.ones((d,d),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def CV_Canny(img):
    return cv2.Canny(img, 100, 200)

def CV_ContourDetection(img):
    dest=255-cv2.Canny(img,100,100,True)
    return (dest) 


def CV_OilPaintingEffect(img, size=7, dynRatio=1):
    """
    Apply oil painting effect to an image.

    Args:
        img (np.array): Input BGR image.
        size (int): Neighborhood size for effect.
        dynRatio (int): Dynamic ratio, higher values increase effect.
        apply (bool)
    Returns:
        np.array: Image with oil painting effect.
    """
    if not hasattr(cv2, 'xphoto') or not hasattr(cv2.xphoto, 'oilPainting'):
        raise ImportError("OpenCV xphoto module or oilPainting function not available. Install opencv-contrib-python.")
    return cv2.xphoto.oilPainting(img, size, dynRatio)

def CV_PointillismEffect(img, dot_radius=5, step=10):
    """
    Apply pointillism effect by drawing colored dots on a white canvas.

    Args:
        img (np.array): Input BGR image.
        dot_radius (int): Radius of dots.
        step (int): Step size between dots.
        apply (bool)
    Returns:
        np.array: Image with pointillism effect.
    """
    height, width = img.shape[:2]
    canvas = 255 * np.ones_like(img)
    for y in range(0, height, step):
        for x in range(0, width, step):
            color = img[y, x].tolist()
            cv2.circle(canvas, (x, y), dot_radius, color, -1, lineType=cv2.LINE_AA)
    return canvas


def CV_AdvancedPointillism(img, num_colors=20, dot_radius=None, step=None):
    """
    Apply advanced pointillism effect by reducing color palette and jittering dot positions.

    Args:
        img (np.array): Input BGR image.
        num_colors (int): Number of colors for palette reduction.
        dot_radius (int): Dot radius; auto-calculated if None.
        step (int): Step between dots; auto-calculated if None.
        apply (bool)
    Returns:
        np.array: Image with advanced pointillism effect.
    """
    h, w = img.shape[:2]
    if dot_radius is None:
        dot_radius = max(3, min(h, w) // 100)
    if step is None:
        step = dot_radius * 2

    pixel_vals = img.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    _, labels, centers = cv2.kmeans(pixel_vals, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    quantized_img = centers[labels].reshape((h, w, 3))
    canvas = 255 * np.ones_like(img)
    rng = np.random.default_rng()

    for y in range(0, h, step):
        for x in range(0, w, step):
            color = quantized_img[y, x].tolist()
            jitter_x = int(rng.integers(-step//3, step//3))
            jitter_y = int(rng.integers(-step//3, step//3))
            cx = np.clip(x + jitter_x, 0, w-1)
            cy = np.clip(y + jitter_y, 0, h-1)
            cv2.circle(canvas, (cx, cy), dot_radius, color, -1, lineType=cv2.LINE_AA)

    return canvas


def CV_AddBackground(img, bg_color=(0, 0, 0)):
    h, w = img.shape[:2]

    if len(img.shape) == 2 or img.shape[2] == 1:  
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  
        alpha = img[:, :, 3] / 255.0
        bgr = img[:, :, :3]
        img = (bgr.astype(float) * alpha[..., np.newaxis] + 
               np.array(bg_color) * (1 - alpha[..., np.newaxis])).astype(np.uint8)    
    ratio = 1920.0 / 1080.0
    new_w = int(ratio * h)
    if new_w < w :
        return img  
    background = np.full((h, new_w, 3), bg_color, dtype=np.uint8)
    x_offset = (new_w - w) // 2
    y_offset = 0
    background[y_offset:y_offset + h, x_offset:x_offset + w] = img
    
    return background


def is_stereo_image(img_stereo, similarity_threshold=0.7):
    img = cv2.cvtColor(img_stereo, cv2.COLOR_BGR2GRAY)
    
    if img is None:
        raise ValueError("Image not found or unsupported file format")

    height, width = img.shape
    ratio = height / width
    
    #print("ratio="+str(ratio))
    
    if width % 2 != 0:
        return False
        #raise ValueError("The image width must be even")
        
    if (ratio > 1.2) :
        return False

    left_img = img[:, :width//2]
    right_img = img[:, width//2:]

    left_norm = (left_img - np.mean(left_img)) / (np.std(left_img) + 1e-10)
    right_norm = (right_img - np.mean(right_img)) / (np.std(right_img) + 1e-10)

    correlation = np.mean(left_norm * right_norm)
    
    #print("Stereo correlation="+str(correlation))

    return correlation > similarity_threshold


def CV_Stereo_Anaglyph_Gray(img_stereo, parallax_offset=0, lim_ratio=2.0):
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


def CV_Stereo_Anaglyph_Color(img_stereo, qStereoImage, parallax_offset=0):
    height, width, _ = img_stereo.shape

    ratio = width / height

    if qStereoImage:
        img_left = img_stereo[:, :width//2, :]
        img_right = img_stereo[:, width//2:, :]
    else:
        img_left = img_stereo
        img_right = img_stereo
        
    if parallax_offset != 0:
        M = np.float32([[1, 0, parallax_offset], [0, 1, 0]])
        img_right = cv2.warpAffine(
            img_right,
            M,
            (img_right.shape[1], img_right.shape[0]),
            borderMode=cv2.BORDER_REPLICATE
        )

    min_height = min(img_left.shape[0], img_right.shape[0])
    min_width = min(img_left.shape[1], img_right.shape[1])
    img_left = img_left[:min_height, :min_width]
    img_right = img_right[:min_height, :min_width]

    anaglyph = np.zeros((min_height, min_width, 3), dtype=img_stereo.dtype)
    anaglyph[:, :, 0] = img_right[:,:,0]   # Blue
    #anaglyph[:, :, 1] = img_left[:,:,1]    # Green
    anaglyph[:, :, 1] = img_right[:,:,1]    # Green
    anaglyph[:, :, 2] = img_left[:,:,2]    # Red
    
    return anaglyph


#------------------------------------------------------------------------------

def is_pixel_down(img, x, y, value):
    if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
        return False
    pixel_value = img[y, x] 
    if len(img.shape) == 3:
        pixel_value = cv2.cvtColor(img[y:y+1, x:x+1], cv2.COLOR_BGR2GRAY)[0, 0]
    return pixel_value < value

def is_pixel_up(img, x, y, value):
    if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
        return False
    pixel_value = img[y, x] 
    if len(img.shape) == 3:
        pixel_value = cv2.cvtColor(img[y:y+1, x:x+1], cv2.COLOR_BGR2GRAY)[0, 0]
    return pixel_value > value

def get_cropped_image(image):
    h, w = image.shape[:2]
    c_left = 0
    c_right = w
        
    c_top = int (h * 0.1) 
    c_bottom = h
    dest = image[c_top:c_bottom, c_left:c_right]
    return (dest)


def num_type_zone(image):
    h, w = image.shape[:2]
    lm = 50
    lx = 9
    ly = 9
    q1 = is_pixel_down(image, lx, ly, lm) and is_pixel_down(image, w-lx, ly, lm) and is_pixel_down(image, w // 2, ly, lm)
    
    q1n = is_pixel_down(image, lx, 200, lm) or is_pixel_down(image,w-lx, 200, lm)  
    
    q1u = True
    q1d = True
    q1n = False
    for i in range(1,w, 10):
        q1u = q1u and is_pixel_down(image, i, 9, lm)
        q1d = q1d and is_pixel_down(image, i, h -9, lm)
        q1n = q1n or is_pixel_up(image, i, h//2, lm) 
              
    q1 = q1u and q1n
    if q1u and q1d :
        q1 = False
     
    if (False):
        print("-------------------------")
        print("q1u="+str(q1u))      
        print("q1n="+str(q1n))
        print("q1d="+str(q1d))
        print("q1="+str(q1))
        
    lm = 240
    ly = 31
    q2 = is_pixel_up(image, 56, ly, lm) and not is_pixel_up(image, 56, ly-10, lm)
    
    ly = 23
    q3 = is_pixel_up(image, 56, ly, lm) and not is_pixel_up(image, 56, ly-10, lm)
    
    q2 = q2 or q3
    

    ly = h - 100
    lm = 240
    q4 = is_pixel_up(image, 90, ly, lm) and is_pixel_up(image, 125, ly, lm)

    return(q1*1+q2*2+q4*4)
    

def get_cropped_image_num(image,num):
    h, w = image.shape[:2]
    c_left = 0
    c_right = w
    c_top = 0
    c_bottom = h
    
    print("num="+str(num))
    
    if (num==1):
        c_top = max(int (h * 0.1),144) 
        c_bottom = h
    if (num==2):
        c_top = 35 
        c_bottom = h -195 * 0
    if (num==3):
        c_top = 35 
        c_bottom = h -195 * 0
        
    if (num==6):
        c_top = 35 
        c_bottom = h -195
        
    dest = image[c_top:c_bottom, c_left:c_right]
    return (dest)

def get_cropped_movie(image):
    h, w = image.shape[:2]
    c_left = 0
    c_right = w
        
    c_top = int (h * 0.055) 
    c_bottom = h - int (h * 0.139)
    
    if math.isclose(float(h) / float(w), 2.11111, rel_tol=1e-4):
        c_top = int (h * 0.07) 
        c_bottom = h - int (h * 0.08)
        
    dest = image[c_top:c_bottom, c_left:c_right]
    return (dest)

#------------------------------------------------------------------------------

def view_picture_zoom(image_path, qAddBackground):
 
    if image_path.lower().endswith(('.avif','.heif')):
        img = cv_load_image_avif(image_path)
    else:    
        img = cv2.imread(image_path)
    
    
    zoom_scale = 1.0
    zoom_min = 1.0
    zoom_max = 15.0
    mouse_x, mouse_y = -1, -1
    height, width = img.shape[:2]
    parallax_offset = -8
    lim_ratio_anaglyph = 2.0
    qLoop = True
    qSharpen = False
    qEnhanceColor = False
    qVibrance = False
    qSaliency = False
    qClache   = False
    qBrightnessContrast = False
    qAdaptativeContrast = False
    qEntropy = False
    qAnaglyph = False
    levelAnaglyph = 0
    qStereoImage = False
    qAutoCrop = False
    qCanny = False
    qDilate = False
    qErode = False
    qPointillismEffect = False
    qOilPaintingEffect = False
    qResizeToWindow = False
    #qAddBackground = False

    
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

   
    num_type_crop = num_type_zone(img)
    qAutoCrop = (num_type_crop>0)
    
    if qAutoCrop:  
        img = get_cropped_image_num(img, num_type_crop)
        height, width = img.shape[:2]
        
        
    
                                 
    ratio = width / height
    
    if is_stereo_image(img):
        qStereoImage = True
        qAnaglyph = True
        width = width // 2
        ratio = width / height
        levelAnaglyph = 1
        parallax_offset = 0
        qAddBackground = False
        
    if qAddBackground and not qAnaglyph :
        img = CV_AddBackground(img)
        height, width = img.shape[:2]
        ratio = width / height
         
    lh = 900
    lw = int(lh * ratio)
    
    if lw > 1920:
        lw = 1910
        lh = int(lw / ratio)
        
        
    window_name = 'Picture Zoom'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1) 
        
    cv2.resizeWindow(window_name, lw, lh)

    screen_width = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) or 1920  
    screen_height = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) or 1080  

    start_x = int((screen_width - lw) / 2)
    start_y = int((screen_height - lh) / 2)
    cv2.moveWindow(window_name, start_x, start_y)
    
    cv2.setMouseCallback(window_name, mouse_callback)
    time.sleep(10/1000)
    
    if height < lh : # Auto qResize
        qResizeToWindow = True

    while qLoop:
        if mouse_x == -1 and mouse_y == -1:
            mouse_x, mouse_y = width // 2, height // 2
            
        if qAnaglyph and levelAnaglyph==1:
            img2 = CV_Stereo_Anaglyph_Color(img, qStereoImage, parallax_offset)
            zoomed_img = get_zoomed_image(img2, zoom_scale, mouse_x, mouse_y)
        else:
            if qResizeToWindow:
                img2 = cv2.resize(img, ( int(lh * ratio), lh), interpolation=cv2.INTER_LINEAR)
                zoomed_img = get_zoomed_image(img2, zoom_scale, mouse_x, mouse_y)
            else :
                zoomed_img = get_zoomed_image(img, zoom_scale, mouse_x, mouse_y)
            
        
            
        if qPointillismEffect : zoomed_img = CV_PointillismEffect(zoomed_img, 7, 10) 
        if qOilPaintingEffect : zoomed_img = CV_OilPaintingEffect(zoomed_img, 3, 1)
        
        if qDilate : zoomed_img = CV_Dilate(zoomed_img)
        if qErode : zoomed_img = CV_Erode(zoomed_img)
        if qCanny : zoomed_img = CV_Canny(zoomed_img)
        if qClache  : zoomed_img = CV_CLAHE(zoomed_img)
        if qSharpen : zoomed_img = CV_Sharpen2d(zoomed_img, 0.1, 0.0,  1)       
        if qEnhanceColor : zoomed_img = CV_EnhanceColor(zoomed_img)
        if qVibrance : zoomed_img = CV_Vibrance2D(zoomed_img)
        if qBrightnessContrast : zoomed_img = CV_AdjustBrightnessContrast(zoomed_img)
        if qAdaptativeContrast : zoomed_img = CV_AdaptativeContrast(zoomed_img)
        if qSaliency : zoomed_img = CV_SaliencyAddWeighted(zoomed_img)
        if qEntropy: zoomed_img = CV_Entropy(zoomed_img)
            
        if qAnaglyph and levelAnaglyph==0:
            zoomed_img = CV_Stereo_Anaglyph_Color(zoomed_img, qStereoImage, parallax_offset)
            

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
        elif key == ord('y'): qCanny = not qCanny
        elif key == ord('D'): qDilate = not qDilate
        elif key == ord('d'): qErode = not qErode
        elif key == ord('p'): qPointillismEffect = not qPointillismEffect
        elif key == ord('o'): qOilPaintingEffect = not qOilPaintingEffect
        elif key == ord('r'): qResizeToWindow = not qResizeToWindow
        
        elif key == ord('.'): zoom_scale = 1.0
        
        elif key == ord('n'): qAnaglyph = not qAnaglyph
        elif key == ord('4'): parallax_offset = parallax_offset - 1
        elif key == ord('6'): parallax_offset = parallax_offset + 1
        
    cv2.destroyAllWindows()
    time.sleep(500/1000)
    
#------------------------------------------------------------------------------

def view_pdf_zoom(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    page_count = doc.page_count

    page_index = 0
    zoom_scale = 1.0
    zoom_min = 1.0
    zoom_max = 15.0
    mouse_x, mouse_y = -1, -1
    qLoop = True
    numEventMouse = 0
    numEvent = 0
    
    qVibrance = False
    qSaliency = False
    qSharpen = False
    qEnhanceColor = False
    qAdaptativeContrast = False
    
    qResizeToWindow = False
    
    qDrawLineOnImage = True
    
    qDilateText = False
    
    window_name = 'PDF Viewer'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    def draw_line_on_image(num_frame, nb_frames,img):
        height, width = img.shape[:2]
        ratio = num_frame / nb_frames
        x = int(ratio * width)    
        image_with_line = img.copy()
        cv2.line(image_with_line,(0, height-9), (x, height-9),(0, 0, 0), 6)
        cv2.line(image_with_line,(0, height-10), (x, height-10),(0, 0, 125), 6)
        cv2.line(image_with_line,(0, height-10), (x, height-10),(0, 0, 255), 3)
        return image_with_line

    def render_page(index):
        page = doc.load_page(index)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat)
        # pixmaps PyMuPDF -> numpy RGB -> BGR pour OpenCV
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    img = render_page(page_index)
    height, width = img.shape[:2]

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
    
    def mouse_click_inside(x, y, x1, y1, x2, y2):
        return x1 <= x <= x2 and y1 <= y <= y2

    def mouse_callback(event, x, y, flags, param):
        nonlocal zoom_scale, mouse_x, mouse_y, qLoop, numEventMouse
        mouse_x, mouse_y = x, y
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags < 0:
                zoom_scale = min(zoom_scale + 0.1, zoom_max)
            else:
                zoom_scale = max(zoom_scale - 0.1, zoom_min)
        if event == cv2.EVENT_RBUTTONDOWN:
            qLoop = False
            
        if event == cv2.EVENT_LBUTTONDOWN:
            numEventMouse = 1
                 

    

    ratio = width / height
    lh = 900
    lw = int(lh * ratio)
    
    if lw > 1920:
        lw = 1910
        lh = int(lw / ratio)
        
        
    cv2.resizeWindow(window_name, lw, lh)
    
    screen_width = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) or 1920  
    screen_height = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) or 1080  
    
    start_x = int((screen_width - lw) / 2)
    start_y = int((screen_height - lh) / 2)

    cv2.moveWindow(window_name, start_x, start_y)
    cv2.setMouseCallback(window_name, mouse_callback)
    time.sleep(10/1000)
        
    while qLoop:
        if mouse_x == -1 and mouse_y == -1:
            mouse_x, mouse_y = width // 2, height // 2
                   
        if qResizeToWindow:
            img = cv2.resize(img, (lw, lh), interpolation=cv2.INTER_LINEAR)            
            
        zoomed_img = get_zoomed_image(img, zoom_scale, mouse_x, mouse_y)
        
        
        if qDilateText :
            zoomed_img = CV_Erode(zoomed_img)
        if qSharpen :
            zoomed_img = CV_Sharpen2d(zoomed_img, 0.1, 0.0,  1)
        if qEnhanceColor : 
            zoomed_img = CV_EnhanceColor(zoomed_img)
        if qAdaptativeContrast :
            zoomed_img = CV_AdaptativeContrast(zoomed_img)
        if qVibrance : 
            zoomed_img = CV_Vibrance2D(zoomed_img)        
        if qSaliency :
            zoomed_img = CV_SaliencyAddWeighted(zoomed_img)
        
        if qDrawLineOnImage :
            zoomed_img=draw_line_on_image(page_index+1, page_count, zoomed_img)
            
        cv2.imshow(window_name, zoomed_img)
        
        if numEventMouse==1:
            numEventMouse = 0
            if qResizeToWindow:
                if mouse_click_inside(mouse_x, mouse_y, lw - int(0.05*lw), 0, lw , lh):
                    numEvent = 3
                if mouse_click_inside(mouse_x, mouse_y, 0, 0, int(0.05*lw), lh):
                    numEvent = 1                
            else :            
                if mouse_click_inside(mouse_x, mouse_y, width - int(0.1*width), 0, width , height):
                    numEvent = 3
                if mouse_click_inside(mouse_x, mouse_y, 0, 0, int(0.1*width), height):
                    numEvent = 1
        
        key = cv2.waitKey(20) & 0xFF
        if key == 27:      # ESC
            break
        elif key == ord(' '):
            numEvent = 3
        elif key == ord('s'):
            path = Path(pdf_path).parent
            new_path = path / "Screenshot"
            new_path.mkdir(parents=True, exist_ok=True)
            date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            outputName = f"PDF_{page_index+1}_{date_time}.jpg"
            outputName = new_path / outputName
            cv2.imwrite(str(outputName), zoomed_img)
        elif key == ord('.'): zoom_scale = 1.0
        elif key == ord('3') or key == ord(' '): numEvent = 3
        elif key == ord('1'): numEvent = 1
        elif key == ord('L'): qDrawLineOnImage = not qDrawLineOnImage
        elif key == ord('x'): qSharpen = not qSharpen
        elif key == ord('e'): qEnhanceColor = not qEnhanceColor
        elif key == ord('c'): qAdaptativeContrast = not qAdaptativeContrast
        elif key == ord('v'): qVibrance = not qVibrance 
        elif key == ord('a'): qSaliency = not qSaliency
        elif key == ord('r'): 
            qResizeToWindow = not qResizeToWindow
            img = render_page(page_index)
        elif key == ord('d'): qDilateText = not qDilateText
        
        if numEvent == 3:
            numEvent = 0
            if page_index < page_count - 1:
                page_index += 1
                img = render_page(page_index)
                height, width = img.shape[:2]
                ratio = width / height
                lh = 900
                lw = int(lh * ratio)
                cv2.resizeWindow(window_name, lw, lh)
                start_x = int((screen_width - lw) / 2)
                start_y = int((screen_height - lh) / 2)
                cv2.moveWindow(window_name, start_x, start_y)
                mouse_x, mouse_y = -1, -1
                zoom_scale = 1.0
                
        if numEvent == 1:
            numEvent = 0
            if page_index > 0:
                page_index -= 1
                img = render_page(page_index)
                height, width = img.shape[:2]
                ratio = width / height
                lh = 900
                lw = int(lh * ratio)
                cv2.resizeWindow(window_name, lw, lh)
                start_x = int((screen_width - lw) / 2)
                start_y = int((screen_height - lh) / 2)
                cv2.moveWindow(window_name, start_x, start_y)
                mouse_x, mouse_y = -1, -1
                zoom_scale = 1.0
        

    cv2.destroyAllWindows()
    time.sleep(500/1000)
    doc.close()


#------------------------------------------------------------------------------

def view_in_mode_opencv(txt_path, font_scale=0.5, line_height=30, max_width=1900):
    if not os.path.isfile(txt_path):
        print(f"File not found : {txt_path}")
        return

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    md = MarkdownIt("commonmark")
    # md = MarkdownIt("commonmark").enable("table").enable("strikethrough")
    tokens = md.parse(text)
    root = SyntaxTreeNode(tokens)  

    font = cv2.FONT_HERSHEY_SIMPLEX

    blocks = []

    def inline_tokens_to_runs(inline_token):
        runs = []
        style_stack = []

        def current_style():
            return {
                "bold": "strong" in style_stack,
                "italic": "em" in style_stack,
                "code": "code" in style_stack,
                "link": "link" in style_stack,
            }

        for t in inline_token.children or []:
            if t.type in ("strong_open", "em_open", "link_open"):
                if t.type == "strong_open":
                    style_stack.append("strong")
                elif t.type == "em_open":
                    style_stack.append("em")
                elif t.type == "link_open":
                    style_stack.append("link")
            elif t.type in ("strong_close", "em_close", "link_close"):
                if t.type == "strong_close" and "strong" in style_stack:
                    style_stack.remove("strong")
                elif t.type == "em_close" and "em" in style_stack:
                    style_stack.remove("em")
                elif t.type == "link_close" and "link" in style_stack:
                    style_stack.remove("link")
            elif t.type == "code_inline":
                st = current_style()
                st["code"] = True
                runs.append((t.content, st))
            elif t.type == "text":
                runs.append((t.content, current_style()))
            elif t.type == "softbreak":
                runs.append(("\n", current_style()))
        return runs

    def walk_block(node, indent_level=0):
        if node.type == "heading":
            level = int(node.attrs.get("level", 1))
            for child in node.children:
                if child.type == "inline":
                    runs = inline_tokens_to_runs(child.token)
                    blocks.append({
                        "type": "heading",
                        "level": level,
                        "runs": runs,
                        "indent": indent_level,
                    })

        elif node.type == "paragraph":
            for child in node.children:
                if child.type == "inline":
                    runs = inline_tokens_to_runs(child.token)
                    blocks.append({
                        "type": "paragraph",
                        "level": 0,
                        "runs": runs,
                        "indent": indent_level,
                    })

        elif node.type in ("bullet_list", "ordered_list"):
            for li in node.children:
                if li.type == "list_item":
                    for ch in li.children:
                        if ch.type == "paragraph":
                            for inl in ch.children:
                                if inl.type == "inline":
                                    runs = inline_tokens_to_runs(inl.token)
                                    blocks.append({
                                        "type": "list_item",
                                        "level": 0,
                                        "runs": runs,
                                        "indent": indent_level + 1,
                                    })
                        else:
                            walk_block(ch, indent_level + 1)

        elif node.type == "fence" or node.type == "code_block":
            code_text = node.token.content if node.token else ""
            runs = [(line, {"bold": False, "italic": False, "code": True, "link": False}) for line in code_text.splitlines()]
            for r in runs:
                blocks.append({
                    "type": "code_block",
                    "level": 0,
                    "runs": [r],
                    "indent": indent_level,
                })

        else:
            for child in node.children:
                walk_block(child, indent_level)

    for child in root.children:
        walk_block(child, indent_level=0)



    wrapped_lines = []  

    for blk in blocks:
        block_type = blk["type"]
        level = blk["level"]
        indent = blk["indent"]
        left_margin = 40 + indent * 50

        words = []  
        for txt, st in blk["runs"]:
            parts = txt.split(" ")
            for i, w in enumerate(parts):
                if w == "":
                    continue
                token = w
                words.append((token + (" " if i < len(parts) - 1 else ""), st))

        if not words:
            wrapped_lines.append({"runs": [], "block_type": block_type, "level": level, "indent": indent})
            continue

        current_runs = []
        current_width = 0

        for word, st in words:
            if block_type == "heading":
                if level == 1:
                    local_scale = font_scale * 1.3
                elif level == 2:
                    local_scale = font_scale * 1.2
                else:
                    local_scale = font_scale * 1.1
            else:
                local_scale = font_scale
                if st.get("bold"):
                    local_scale *= 1.2

            (w_px, _), _ = cv2.getTextSize(word, font, local_scale, 1)
            if left_margin + current_width + w_px > max_width - 20 and current_runs:
                wrapped_lines.append({
                    "runs": current_runs,
                    "block_type": block_type,
                    "level": level,
                    "indent": indent,
                })
                current_runs = []
                current_width = 0
            current_runs.append((word, st))
            current_width += w_px

        if current_runs:
            wrapped_lines.append({
                "runs": current_runs,
                "block_type": block_type,
                "level": level,
                "indent": indent,
            })

    if not wrapped_lines:
        print("Nothing to render in Markdown.")
        return

    ratio_screen = 1920 / 1080
    img_height = max(line_height * (len(wrapped_lines) + 4), 1080)
    #img_width = max_width
    img_width = int(img_height * ratio_screen)
    
    img = np.full((img_height, img_width, 3), 30, dtype=np.uint8)
    
    y = int (line_height * 1.5)
    for line in wrapped_lines:
        runs = line["runs"]
        block_type = line["block_type"]
        level = line["level"]
        indent = line["indent"]
        x = 20 + indent * 50

        # bullet 
        if block_type == "list_item":
            cv2.circle(img, (x - 10, y - 6), 3, (255, 255, 255), -1)

        for txt, st in runs:
            if block_type == "heading":
                if level == 1:
                    local_scale = font_scale * 1.3
                elif level == 2:
                    local_scale = font_scale * 1.2
                else:
                    local_scale = font_scale * 1.1
                color = (250, 240, 240)  
                thickness = 2
            elif block_type == "code_block" or st.get("code"):
                local_scale = font_scale * 0.9
                color = (60, 60, 60)
                thickness = 1
            elif st.get("link"):
                local_scale = font_scale
                color = (200, 0, 0)  
                thickness = 1
            elif st.get("bold"):
                local_scale = font_scale * 1.1
                color = (255, 255, 255)
                thickness = 1
            else:
                local_scale = font_scale
                color = (255, 255, 255)
                thickness = 1

            (w_px, _), _ = cv2.getTextSize(txt, font, local_scale, thickness)
            
            cv2.putText(img, txt, (x+1, y+1), font, local_scale, (0,0,0), thickness, cv2.LINE_AA)
            cv2.putText(img, txt, (x, y), font, local_scale, color, thickness, cv2.LINE_AA)
            x += w_px

        y += line_height

    zoom_scale = 1.0
    zoom_min = 1.0
    zoom_max = 15.0
    mouse_x, mouse_y = -1, -1
    qLoop = True
    window_name = 'MD Viewer'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    h, w = img.shape[:2]
    ratio = w / h
    lh = 900
    lw = int(lh * ratio)
    if lw > 1920:
        lw = 1800
        lh = int(lw / ratio)
    cv2.resizeWindow(window_name, lw, lh)

    screen_width = 1920
    screen_height = 1080
    start_x = int((screen_width - lw) / 2)
    start_y = int((screen_height - lh) / 2)
    cv2.moveWindow(window_name, start_x, start_y)

    def get_zoomed_image(image, scale, cx, cy):
        h0, w0 = image.shape[:2]
        new_w = int(w0 / scale)
        new_h = int(h0 / scale)
        left = max(cx - new_w // 2, 0)
        right = min(cx + new_w // 2, w0)
        top = max(cy - new_h // 2, 0)
        bottom = min(cy + new_h // 2, h0)
        if right - left < new_w:
            if left == 0:
                right = new_w
            elif right == w0:
                left = w0 - new_w
        if bottom - top < new_h:
            if top == 0:
                bottom = new_h
            elif bottom == h0:
                top = h0 - new_h
        cropped = image[top:bottom, left:right]
        zoomed = cv2.resize(cropped, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return zoomed

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

    cv2.setMouseCallback(window_name, mouse_callback)
    time.sleep(10/1000)

    while qLoop:
        if mouse_x == -1 and mouse_y == -1:
            mouse_x, mouse_y = w // 2, h // 2
        zoomed_img = get_zoomed_image(img, zoom_scale, mouse_x, mouse_y)
        cv2.imshow(window_name, zoomed_img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        elif key == ord('.'):
            zoom_scale = 1.0
        elif key == ord('s'):
            path = Path(txt_path).parent
            new_path = path / "Screenshot"
            new_path.mkdir(parents=True, exist_ok=True)
            date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            outputName = new_path / f"MD_{date_time}.jpg"
            cv2.imwrite(str(outputName), zoomed_img)

    cv2.destroyAllWindows()
    time.sleep(500/1000)
    
    
#------------------------------------------------------------------------------
    
def view_in_mode_txt(txt_path, font_scale=0.5, line_height=18, max_width=1600):
    
    import tkinter as tkS
    if not os.path.isfile(txt_path):
        print(f"File not found : {txt_path}")
        return

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()


    md = MarkdownIt("commonmark").enable("table").enable("strikethrough")
    tokens = md.parse(text)
    root = SyntaxTreeNode(tokens)  


    root_tkS = tkS.Tk()
    root_tkS.title(f"Markdown viewer - {os.path.basename(txt_path)}")

    bg_color = "#ffffff"
    fg_color = "#24292e"
    code_bg = "#f6f8fa"
    blockquote_color = "#6a737d"
    table_border_color = "#d0d7de"
    link_color = "#0969da"
      
    bg_color = "#000000"      # noir
    fg_color = "#FFFFFF"      # blanc
    code_bg = "#f6f8fa"
    blockquote_color = "#6a737d"
    table_border_color = "#555555"
    link_color = "#4ea3ff"    # bleu clair lisible sur fond sombre
    
    
    approx_char_width = int(max_width / 8)

    text_widget = scrolledtext.ScrolledText(
        root_tkS,
        wrap=tkS.WORD,
        width=approx_char_width,
        height=40,
        bg=bg_color,
        fg=fg_color,
        insertbackground=fg_color,
        borderwidth=0
    )
    text_widget.pack(fill=tkS.BOTH, expand=True)

    text_widget.config(state=tkS.NORMAL)

    base_font_size = int(12 * font_scale * 2) or 11

    # Styles
    text_widget.tag_configure("paragraph", font=("TkDefaultFont", base_font_size))
    text_widget.tag_configure("bold", font=("TkDefaultFont", base_font_size, "bold"))
    text_widget.tag_configure("italic", font=("TkDefaultFont", base_font_size, "italic"))
    text_widget.tag_configure("strike", overstrike=1)
    text_widget.tag_configure("code", font=("Courier New", base_font_size), background=code_bg)
    text_widget.tag_configure("link", foreground=link_color, underline=1)
    text_widget.tag_configure("h1",
                              font=("TkDefaultFont", int(base_font_size * 1.9), "bold"),
                              spacing1=10, spacing3=6)
    text_widget.tag_configure("h2",
                              font=("TkDefaultFont", int(base_font_size * 1.6), "bold"),
                              spacing1=8, spacing3=4)
    text_widget.tag_configure("h3",
                              font=("TkDefaultFont", int(base_font_size * 1.3), "bold"),
                              spacing1=6, spacing3=2)

    text_widget.tag_configure("list_item", lmargin1=20, lmargin2=40)
    text_widget.tag_configure("blockquote",
                              lmargin1=20,
                              lmargin2=40,
                              foreground=blockquote_color)
    text_widget.tag_configure("table_header",
                              font=("TkDefaultFont", base_font_size, "bold"))
    text_widget.tag_configure("table_border",
                              foreground=table_border_color)
    text_widget.tag_configure("table_cell",
                              font=("TkDefaultFont", base_font_size))

    def insert_with_tags(txt, tags):
        start = text_widget.index(tkS.INSERT)
        text_widget.insert(tkS.INSERT, txt)
        end = text_widget.index(tkS.INSERT)
        for tag in tags:
            text_widget.tag_add(tag, start, end)

    def inline_tokens_to_segments(inline_token):
        segments = []
        style_stack = []

        def current_tags():
            tags = ["paragraph"]
            if "strong" in style_stack:
                tags.append("bold")
            if "em" in style_stack:
                tags.append("italic")
            if "code" in style_stack:
                tags.append("code")
            if "link" in style_stack:
                tags.append("link")
            if "strike" in style_stack:
                tags.append("strike")
            return tags

        for t in inline_token.children or []:
            if t.type in ("strong_open", "em_open", "link_open", "s_open"):
                if t.type == "strong_open":
                    style_stack.append("strong")
                elif t.type == "em_open":
                    style_stack.append("em")
                elif t.type == "link_open":
                    style_stack.append("link")
                elif t.type == "s_open":
                    style_stack.append("strike")
            elif t.type in ("strong_close", "em_close", "link_close", "s_close"):
                if t.type == "strong_close" and "strong" in style_stack:
                    style_stack.remove("strong")
                elif t.type == "em_close" and "em" in style_stack:
                    style_stack.remove("em")
                elif t.type == "link_close" and "link" in style_stack:
                    style_stack.remove("link")
                elif t.type == "s_close" and "strike" in style_stack:
                    style_stack.remove("strike")
            elif t.type == "code_inline":
                style_stack.append("code")
                segments.append((t.content, current_tags()))
                style_stack.remove("code")
            elif t.type == "text":
                segments.append((t.content, current_tags()))
            elif t.type == "softbreak":
                segments.append(("\n", current_tags()))
            elif t.type == "hardbreak":
                segments.append(("\n", current_tags()))
        return segments

    def render_table(table_node):
        rows = []
        header_row = None

        def collect_row(tr_node):
            cells = []
            for cell in tr_node.children:
                if cell.type in ("th", "td"):
                    cell_text = ""
                    for ch in cell.children:
                        if ch.type == "inline":
                            segs = inline_tokens_to_segments(ch.token)
                            cell_text += "".join(s[0] for s in segs)
                    cells.append(cell_text.strip())
            return cells

        for child in table_node.children:
            if child.type == "thead":
                for tr in child.children:
                    if tr.type == "tr":
                        header_row = collect_row(tr)
            elif child.type == "tbody":
                for tr in child.children:
                    if tr.type == "tr":
                        rows.append(collect_row(tr))

        if header_row is None and rows:
            header_row = rows[0]
            rows = rows[1:]

        if not header_row:
            return

        col_count = len(header_row)
        widths = [len(h) for h in header_row]
        for r in rows:
            for i, c in enumerate(r):
                if i < col_count:
                    widths[i] = max(widths[i], len(c))

        def format_row(cells):
            padded = []
            for i in range(col_count):
                txt = cells[i] if i < len(cells) else ""
                padded.append(" " + txt.ljust(widths[i]) + " ")
            return "|".join(padded)

        border_line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

        insert_with_tags(border_line + "\n", ["table_border"])
        insert_with_tags("|", ["table_border"])
        for i, cell in enumerate(header_row):
            cell_text = " " + cell.ljust(widths[i]) + " "
            insert_with_tags(cell_text, ["table_header", "table_cell"])
            insert_with_tags("|", ["table_border"])
        text_widget.insert(tkS.INSERT, "\n")

        insert_with_tags(border_line + "\n", ["table_border"])

        for r in rows:
            row_text = format_row(r)
            insert_with_tags("|", ["table_border"])
            cur_idx = 0
            for i in range(col_count):
                cell_content = " " + (r[i] if i < len(r) else "").ljust(widths[i]) + " "
                insert_with_tags(cell_content, ["table_cell"])
                insert_with_tags("|", ["table_border"])
                cur_idx += len(cell_content) + 1
            text_widget.insert(tkS.INSERT, "\n")
        insert_with_tags(border_line + "\n\n", ["table_border"])

 
    def handle_block(node, indent_level=0):
        if node.type == "heading":
            level = int(node.attrs.get("level", 1))
            tag = "h1" if level == 1 else "h2" if level == 2 else "h3"
            for child in node.children:
                if child.type == "inline":
                    segs = inline_tokens_to_segments(child.token)
                    for txt, tags in segs:
                        insert_with_tags(txt, tags + [tag])
            text_widget.insert(tkS.INSERT, "\n\n")

        elif node.type == "paragraph":
            for child in node.children:
                if child.type == "inline":
                    segs = inline_tokens_to_segments(child.token)
                    for txt, tags in segs:
                        insert_with_tags(txt, tags)
            text_widget.insert(tkS.INSERT, "\n\n")

        elif node.type in ("bullet_list", "ordered_list"):
            for li in node.children:
                if li.type == "list_item":
                    start = text_widget.index(tkS.INSERT)
                    text_widget.insert(tkS.INSERT, "• ")
                    text_widget.tag_add("list_item", start, text_widget.index(tk.INSERT))
                    for ch in li.children:
                        if ch.type == "paragraph":
                            for inl in ch.children:
                                if inl.type == "inline":
                                    segs = inline_tokens_to_segments(inl.token)
                                    for txt, tags in segs:
                                        insert_with_tags(txt, tags + ["list_item"])
                        else:
                            handle_block(ch, indent_level + 1)
                    text_widget.insert(tkS.INSERT, "\n")
            text_widget.insert(tkS.INSERT, "\n")

        elif node.type in ("fence", "code_block"):
            code_text = node.token.content if node.token else ""
            for line in code_text.splitlines():
                insert_with_tags(line + "\n", ["code"])
            text_widget.insert(tkS.INSERT, "\n")

        elif node.type == "blockquote":
            before = text_widget.index(tkS.INSERT)
            for child in node.children:
                handle_block(child, indent_level + 1)
            after = text_widget.index(tkS.INSERT)
            text_widget.tag_add("blockquote", before, after)

        elif node.type == "table":
            render_table(node)

        else:
            for child in node.children:
                handle_block(child, indent_level)

    for child in root.children:
        handle_block(child, indent_level=0)

    text_widget.config(state=tkS.DISABLED)


    def _on_mousewheel(event):
        text_widget.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"
        
    #window_closed = {"done": False}

    def _close_window(event=None):
       #if not window_closed["done"]:
           #window_closed["done"] = True
           root_tkS.destroy()
           time.sleep(500/1000)

    text_widget.bind("<Button-3>", _close_window)
    root_tkS.protocol("WM_DELETE_WINDOW", _close_window)
    #root_tkS.bind("<Escape>", _close_window)    
    root_tkS.mainloop()

    
    
#------------------------------------------------------------------------------

def play_video_with_seek_and_pause(video_path, qAddBackground):
    zoom_scale = 1.0
    zoom_min = 1.0
    zoom_max = 15.0
    mouse_x, mouse_y = -1, -1
    parallax_offset = -8
    lim_ratio_anaglyph = 2.0
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
    qAnaglyph = False
    qStereoImage = False
    levelAnaglyph = 0
    qAutoCrop = False
    qResizeToWindow = False
    qCanny = False
    qDilate = False
    qErode = False
    #qAddBackground = True
    
    disp_w, disp_h = None, None
    
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
        
        
        if event == cv2.EVENT_RBUTTONDOWN:
            qLoop = False
                
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
            
            #if disp_w is None or disp_h is None:
            #    return
            
            if mouse_click_inside(mouse_x, mouse_y, 0, 0, disp_w, int(0.1 * disp_h)):
                paused = not paused
            else:
                # x par rapport à la largeur affichée -> frame vidéo
                clicked_frame = int((x / disp_w) * frame_count)
                clicked_frame = max(0, min(clicked_frame, frame_count - 1))
                current_frame = clicked_frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                paused = False
                
            

            
    
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
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
    ret, frame = cap.read()   
    ratio = width / height
            
    qAutoCrop = is_pixel_down(frame, 3, 3, 15) and is_pixel_down(frame, width-3, 3, 15) and is_pixel_down(frame, width // 2, 3, 15)
    qAutoCrop = qAutoCrop and (height > width)
    
    if qAutoCrop:
        img = get_cropped_movie(frame)
        height, width = img.shape[:2]
        ratio = width / height
    
    
    if is_stereo_image(frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min (60 * 25, frame_count))
        ret, frame = cap.read()
        if is_stereo_image(frame):
            qAnaglyph = True
            qStereoImage = True
            width = width // 2
            ratio = width / height
            parallax_offset = 0
            qAddBackground = False
            levelAnaglyph = 1
        

    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ret, frame = cap.read()
    
    if qAddBackground and not qAnaglyph :
        frame = CV_AddBackground(frame)
        height, width = frame.shape[:2]
        ratio = width / height
    
    window_name = 'Movie Player'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    lh = 900
    lw = int(lh * ratio)
    
    if lw > 1920:
        lw = 1910
        lh = int(lw / ratio)
    
    cv2.resizeWindow(window_name, lw, lh)
    
    screen_width = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) or 1920  
    screen_height = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) or 1080  

    start_x = int((screen_width - lw) / 2)
    start_y = int((screen_height - lh) / 2)
        
    time.sleep(10/1000)
    cv2.moveWindow(window_name, start_x, start_y)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    if height < lh : # Auto qResize
        qResizeToWindow = True
        
    pausedLevel = 0
    qPostTraitementFrame = True

    while qLoop:
        if not paused:
            qPostTraitementFrame = True
            pausedLevel=0
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
            if pausedLevel==0 :
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if not ret:
                    break
                pausedLevel=1
                qPostTraitementFrame = True

        if qPostTraitementFrame :
            if qAutoCrop:
                frame = get_cropped_movie(frame)
                            
            if qAddBackground:
                frame = CV_AddBackground(frame)
                
            if qResizeToWindow:
                lw = int(lh * ratio)
                frame = cv2.resize(frame, (lw, lh), interpolation=cv2.INTER_LINEAR)      
            
        if pausedLevel==1 : 
            qPostTraitementFrame = False
            

        if qAnaglyph and levelAnaglyph==1:
            img2 = CV_Stereo_Anaglyph_Color(frame, qStereoImage, parallax_offset)
            zoomed_img = get_zoomed_image(img2, zoom_scale, mouse_x, mouse_y)
        else:
            zoomed_img = get_zoomed_image(frame, zoom_scale, mouse_x, mouse_y)
              
            
        if qDilate : zoomed_img = CV_Dilate(zoomed_img)
        if qErode : zoomed_img = CV_Erode(zoomed_img)
        if qCanny : zoomed_img = CV_Canny(zoomed_img) 
        if qClache  : zoomed_img = CV_CLAHE(zoomed_img)
        if qSharpen : zoomed_img = CV_Sharpen2d(zoomed_img, 0.1, 0.0,  1)
        if qEnhanceColor : zoomed_img = CV_EnhanceColor(zoomed_img)
        if qVibrance : zoomed_img = CV_Vibrance2D(zoomed_img)
        if qBrightnessContrast : zoomed_img = CV_AdjustBrightnessContrast(zoomed_img)
        if qAdaptativeContrast : zoomed_img = CV_AdaptativeContrast(zoomed_img)
        if qSaliency : zoomed_img = CV_SaliencyAddWeighted(zoomed_img)
            
        if qEntropy: zoomed_img = CV_Entropy(zoomed_img)
            
        if qAnaglyph and levelAnaglyph==0:
            zoomed_img = CV_Stereo_Anaglyph_Color(zoomed_img, qStereoImage, parallax_offset)
            
        disp_h, disp_w = zoomed_img.shape[:2]
        
        if qDrawLineOnImage :
            zoomed_img=draw_line_on_image(current_frame, frame_count, zoomed_img)
            
            
        cv2.imshow('Movie Player', zoomed_img)
        
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == 27:  
            break
        elif key == ord(' '): 
            paused = not paused
            #pausedLevel = 0
        elif key == ord('2'): fps = fps_movie
        elif key == ord('1'): fps = max ( 1, fps // 2)
        elif key == ord('2'): fps = fps_movie
        elif key == ord('3'): fps = fps * 2
        elif key == ord('+'): 
            current_frame = current_frame + 1
            pausedLevel = 0
        elif key == ord('-'): 
            current_frame = current_frame - 1
            pausedLevel = 0
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
        elif key == ord('y'): qCanny = not qCanny
        elif key == ord('D'): qDilate = not qDilate
        elif key == ord('d'): qErode = not qErode
        
        elif key == ord('t'): qEntropy = not qEntropy
        elif key == ord('.'): zoom_scale = 1.
        
        elif key == ord('r'): qResizeToWindow = not qResizeToWindow
        
        elif key == ord('n'): qAnaglyph = not qAnaglyph
        elif key == ord('4'): parallax_offset = parallax_offset - 1
        elif key == ord('6'): parallax_offset = parallax_offset + 1
                    
    cap.release()
    cv2.destroyAllWindows()
    time.sleep(500/1000)


#------------------------------------------------------------------------------


def play_audio_with_seek_and_waveform(audio_path):
    if audio_path.lower().endswith(".mp3"):
        audio_info = MP3(audio_path)
        duration_sec = float(audio_info.info.length)
    else:
        audio_info = WAVE(audio_path)
        duration_sec = float(audio_info.info.length)

    audio_seg = AudioSegment.from_file(audio_path)
    samples = np.array(audio_seg.get_array_of_samples())
    if audio_seg.channels > 1:
        samples = samples.reshape((-1, audio_seg.channels)).mean(axis=1)

    samples = samples.astype(np.float32)
    samples /= np.max(np.abs(samples)) + 1e-9

    sr = audio_seg.frame_rate

    width = 1800
    h_wave = 450
    h_spec = 450
    height = h_wave + h_spec

    wave_img = np.zeros((h_wave, width, 3), dtype=np.uint8)
    step = max(1, int(len(samples) / width))
    mid_y = h_wave // 2
    amp = int(h_wave * 0.4)

    for x in range(width):
        idx = x * step
        if idx >= len(samples):
            break
        val = samples[idx]
        y = int(val * amp)
        cv2.line(wave_img,
                 (x, mid_y - y),
                 (x, mid_y + y),
                 (255, 255, 255), 1)

    nperseg = 1024
    noverlap = nperseg // 2
    freqs, times, Sxx = spectrogram(samples, fs=sr, nperseg=nperseg, noverlap=noverlap)  
    Sxx = np.abs(Sxx)
    Sxx = 10 * np.log10(Sxx + 1e-9)

    S_min, S_max = Sxx.min(), Sxx.max()
    S_norm = (Sxx - S_min) / (S_max - S_min + 1e-9)
    spec_img = (S_norm * 255).astype(np.uint8)

    spec_img = cv2.resize(spec_img, (width, h_spec), interpolation=cv2.INTER_LINEAR)
    spec_img = cv2.flip(spec_img, 0)  # 0 Hz en bas

    spec_img_color = cv2.applyColorMap(spec_img, cv2.COLORMAP_JET)


    pygame.mixer.init(frequency=audio_seg.frame_rate,
                      channels=audio_seg.channels)
    pygame.mixer.music.load(audio_path)

    window_name = "Audio Player"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(window_name, width, height)
    
    
    screen_width = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) or 1920  
    screen_height = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) or 1080  

    start_x = int((screen_width - width) / 2)
    start_y = int((screen_height - height) / 2)
    cv2.moveWindow(window_name, start_x, start_y)
    

    qLoop = True
    paused = False
    current_time = 0.0  
    base_offset = 0.0   

    def seek_to_time(t_sec):
        nonlocal current_time, base_offset, paused
        t_sec = max(0.0, min(duration_sec, t_sec))
        base_offset = t_sec
        current_time = t_sec
        pygame.mixer.music.play(start=t_sec)
        paused = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal qLoop
        if event == cv2.EVENT_LBUTTONDOWN and duration_sec > 0:
            ratio = x / float(width)
            ratio = max(0.0, min(1.0, ratio))
            t_sec = ratio * duration_sec
            seek_to_time(t_sec)
        if event == cv2.EVENT_RBUTTONDOWN:
            qLoop = False

    cv2.setMouseCallback(window_name, mouse_callback)

    pygame.mixer.music.play()
    clock = pygame.time.Clock()

    while qLoop:
        if not paused:
            pos_ms = pygame.mixer.music.get_pos()
            if pos_ms < 0:
                qLoop = False
                break
            current_time = base_offset + pos_ms / 1000.0

        frame = np.zeros((height, width, 3), dtype=np.uint8)

        frame[0:h_wave, :, :] = wave_img

        frame[h_wave:h_wave + h_spec, :, :] = spec_img_color

        ratio = current_time / duration_sec if duration_sec > 0 else 0.0
        ratio = max(0.0, min(1.0, ratio))
        x = int(ratio * width)
        cv2.line(frame, (x, 0), (x, h_wave - 1), (0, 0, 255), 2)
        cv2.line(frame, (x, h_wave), (x, height - 1), (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            qLoop = False
        elif key == ord(' '):
            if paused:
                pygame.mixer.music.unpause()
            else:
                pygame.mixer.music.pause()
            paused = not paused
        elif key == ord('.'):
            seek_to_time(0.0)
        elif key == ord('s'):
            path = Path(audio_path).parent
            new_path = path / "Screenshot"
            new_path.mkdir(parents=True, exist_ok=True)
            date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            outputName = f"WaveFFT_{date_time}.jpg"
            outputName = new_path / outputName
            cv2.imwrite(str(outputName), frame)

        clock.tick(60)

    pygame.mixer.music.stop()
    cv2.destroyWindow(window_name)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def sub_is_frame_black(frame, threshold=10):
    return frame.mean() < threshold


def sub_get_first_non_black_frame(video_path, max_frames_to_check=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return Image.new("RGB", (320, 240), color="black")
    frame_number = 0
    while frame_number < max_frames_to_check:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if not sub_is_frame_black(frame, 50):
            cap.release()
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_number += 1
    cap.release()
    return Image.new("RGB", (320, 240), color="black")

def sub_get_non_black_frames_composed(video_path, num_frames=4):
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
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    w0, h0 = img.size
    frame_idx = 0
    while len(frames_collected) < num_frames and frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            break
        if not sub_is_frame_black(frame):
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

def sub_convert_and_resize_image(img_path, output_dir, format, quality=90, height=None):
    try:
        img = Image.open(img_path)
        
        #img = np.array(img)
        #if is_stereo_image(img):
        #    img = CV_Stereo_Anaglyph_Color(img)
            
        if height is not None:
            wpercent = (height / float(img.size[1]))
            width = int(float(img.size[0]) * wpercent)
            img = img.resize((width, height),Image.LANCZOS)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lastExt = os.path.splitext(os.path.basename(img_path))[1]
        output_path = os.path.join(output_dir, f"{base_name}{lastExt}.{format.lower()}")
        if format == "HEIF":
            img.save(output_path, format="HEIF", quality=quality)
        elif format == "AVIF":
            img.save(output_path, format="AVIF", quality=quality)
        else:
            raise ValueError("Unsupported format")
        return output_path
    except Exception as e:
        return f"Error for {img_path}: {e}"

def sub_get_grid_preview_from_pdf(pdf_path, dpi=200, grid_size=3):
    pdf_document = fitz.open(pdf_path)
    page_count = pdf_document.page_count  
    
    if page_count < grid_size * grid_size:
        page = pdf_document.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)  
        pdf_document.close()
        return img
    
    n_tiles = grid_size * grid_size  
    
    indices = []
    for i in range(n_tiles):
        t = i / (n_tiles - 1)
        idx = int(round(t * (page_count - 1)))
        indices.append(idx)
    
    page_images = []
    for idx in indices:
        page = pdf_document.load_page(idx)  
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        page_images.append(img)
    
    pdf_document.close()
    
    tile_w, tile_h = page_images[0].size
    page_images = [im.resize((tile_w, tile_h), Image.LANCZOS) for im in page_images]
    
    grid_w = tile_w * grid_size
    grid_h = tile_h * grid_size
    grid_img = Image.new("RGB", (grid_w, grid_h))
    
    for i, im in enumerate(page_images):
        row = i // grid_size
        col = i % grid_size
        x = col * tile_w
        y = row * tile_h
        grid_img.paste(im, (x, y))
    return grid_img


def sub_convert_and_resize_pdf_image(file_path, output_dir, format, quality=90, height=None):
    try:
        img = sub_get_grid_preview_from_pdf(file_path)
        if height is not None:
            wpercent = (height / float(img.size[1]))
            width = int(float(img.size[0]) * wpercent)
            img = img.resize((width, height),Image.LANCZOS)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        lastExt = os.path.splitext(os.path.basename(file_path))[1]
        output_path = os.path.join(output_dir, f"{base_name}{lastExt}.{format.lower()}")
        if format == "HEIF":
            img.save(output_path, format="HEIF", quality=quality)
        elif format == "AVIF":
            img.save(output_path, format="AVIF", quality=quality)
        else:
            raise ValueError("Unsupported format")
        return output_path
    except Exception as e:
        return f"Error for {file_path}: {e}"


def sub_process_files(name_file_zip, directory, format, workers, quality=90, height=None, process_videos=False, mode=2):
    output_dir = os.path.join(directory, 'TMP')
    os.makedirs(output_dir, exist_ok=True)

    # PICTURES TRAITEMENT
    jpg_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTENSIONS)]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(sub_convert_and_resize_image, f, output_dir, format, quality, height) for f in jpg_files]
        for future in tqdm(futures, total=len(futures), desc=f"Picture thumbnail processing"):
            results.append(future.result())

    # PDFs TRAITEMENT
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(PDF_EXTENSIONS)]
    pdf_thumbs = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(sub_convert_and_resize_pdf_image, pf, output_dir, format, quality, height) for pf in pdf_files]
        for future in tqdm(futures, total=len(futures), desc=f"PDF thumbnail processing"):
            pdf_thumbs.append(future.result())

    # VIDEOs TRAITEMENT
    video_thumbs = []
    if process_videos:
        video_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(VIDEO_EXTENSIONS)]

        def process_single_video(video_file):
            if mode == 2:
                thumb_img = sub_get_non_black_frames_composed(video_file, 4)
            elif mode == 3:
                thumb_img = sub_get_non_black_frames_composed(video_file, 9)
            else:
                thumb_img = sub_get_first_non_black_frame(video_file, 120)

            base_name = os.path.splitext(os.path.basename(video_file))[0]
            lastExt = os.path.splitext(os.path.basename(video_file))[1]
            thumb_path = os.path.join(output_dir, f"{base_name}{lastExt}.{format.lower()}")

            # Enregistrement de la miniature selon le format
            if format == "HEIF":
                thumb_img.save(thumb_path, format="HEIF", quality=quality)
            elif format == "AVIF":
                thumb_img.save(thumb_path, format="AVIF", quality=quality)
            else:
                thumb_img.save(thumb_path, format="JPEG", quality=quality)

            return thumb_path

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_single_video, vf) for vf in video_files]
            for future in tqdm(futures, total=len(futures), desc="Video thumbnail processing"):
                video_thumbs.append(future.result())

    # BUILD ZIP
    zip_path = os.path.join(directory, name_file_zip)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in results + video_thumbs + pdf_thumbs:
            if not file_path.startswith("Erreur"):
                zipf.write(file_path, os.path.basename(file_path))
    shutil.rmtree(output_dir)
    return results + video_thumbs + pdf_thumbs, zip_path


#------------------------------------------------------------------------------

class Slideshow:
    def __init__(self, master, directory, panel_cols, panel_rows, mode, workers, qcache=False,thumb_format="PNG", thumbSizeCache=540, SortFiles="NAME", qModeSoftwareView=True):
        self.master = master
        self.directory = directory
        self.panel_cols = panel_cols
        self.panel_rows = panel_rows
        self.mode = mode
        self.workers = workers
        self.qcache = qcache
        self.thumbSizeCache = thumbSizeCache
        self.qModeSoftwareView = qModeSoftwareView
        self.qModeBackground = False
        self.thumb_format = thumb_format.upper()
        self.panel_step = self.panel_cols * self.panel_rows
        self.current_image = 0
        self.images = self.get_images(SortFiles)
        self.nb_media = len(self.images)
        self.best_grid()
        self.image_refs = []
        PathW = os.path.dirname(sys.argv[0])
        self.audio_placeholder_img = Image.open(PathW + "/audio.jpg")
        self.txt_placeholder_img = Image.open(PathW + "/TXT.jpg")
        self.md_placeholder_img = Image.open(PathW + "/MD.jpg")
        
            
        self.cache_zip_path = None
        if self.qcache:
            fileZipCache = os.path.join(self.directory, "cache_thumbs.zip")
            if os.path.isfile(fileZipCache):
                self.init_video_cache_zip1()
            else :
                self.init_video_cache_zip2()
                self.init_video_cache_zip1()
        
        self.setup_ui()
        self.update_clock()
        self.master.bind("<Escape>", self.exit_app_key)
        self.master.bind("<space>", lambda e: self.next_image())
        self.master.bind("<Left>", lambda e: self.prev_image())
        self.master.bind("<Right>", lambda e: self.next_image())
        
        self.master.bind("m", self.mode_soft_app_key)
        
        self.master.bind("b", self.mode_Background_key)
        
        self.master.bind("<Button-1>", self.on_left_click)


    def init_video_cache_zip1(self):
        for file_path in self.images:
            if file_path.lower().endswith(VIDEO_EXTENSIONS+IMAGE_EXTENSIONS+PDF_EXTENSIONS):
                self.cache_zip_path = get_cache_zip_path(file_path)
                break
        #if self.cache_zip_path:
        #    if has_cache_config_changed_zip(self.cache_zip_path, self.panel_cols, self.panel_rows, self.mode, self.thumb_format):
        #        if os.path.isfile(self.cache_zip_path):
        #            os.remove(self.cache_zip_path)
        #        save_cache_config_to_zip(self.cache_zip_path, self.panel_cols, self.panel_rows, self.mode, self.thumb_format)

    def init_video_cache_zip2(self):        
        sub_process_files(  
            "cache_thumbs.zip",
            self.directory,
            self.thumb_format,
            self.workers,
            90,
            self.thumbSizeCache,
            True,
            self.mode)
  

    def get_cached_or_generate_video_thumb(self, video_path):
        if not self.qcache:
            return self.make_video_thumb(video_path)
        zip_path = get_cache_zip_path(video_path)
        #if has_cache_config_changed_zip(zip_path, self.panel_cols, self.panel_rows, self.mode, self.thumb_format):
        #    if os.path.isfile(zip_path):
        #        os.remove(zip_path)
        #    save_cache_config_to_zip(zip_path, self.panel_cols, self.panel_rows, self.mode, self.thumb_format)
        thumb_name = os.path.basename(video_path) + "." + self.thumb_format.lower()
        cached_img = load_image_from_zip(zip_path, thumb_name)
        if cached_img:
            return cached_img
        img = self.make_video_thumb(video_path)
        save_image_to_zip(zip_path, thumb_name, img, self.thumb_format)
        return img
    
    def get_cached_or_generate_pdf_thumb(self, pdf_path):
        if not self.qcache:
            return self.make_pdf_thumb(pdf_path)
        zip_path = get_cache_zip_path(pdf_path)

        thumb_name = os.path.basename(pdf_path) + "." + self.thumb_format.lower()
        cached_img = load_image_from_zip(zip_path, thumb_name)
        if cached_img:
            return cached_img
        img = self.make_pdf_thumb(pdf_path)
        save_image_to_zip(zip_path, thumb_name, img, self.thumb_format)
        return img
   
    
    def get_cached_or_generate_picture_thumb(self, picture_path):
        if not self.qcache:
            return self.make_picture_thumb(picture_path)
        zip_path = get_cache_zip_path(picture_path)
        #if has_cache_config_changed_zip(zip_path, self.panel_cols, self.panel_rows, self.mode, self.thumb_format):
        #    if os.path.isfile(zip_path):
        #        os.remove(zip_path)
        #    save_cache_config_to_zip(zip_path, self.panel_cols, self.panel_rows, self.mode, self.thumb_format)
        thumb_name = os.path.basename(picture_path) + "." + self.thumb_format.lower()
        cached_img = load_image_from_zip(zip_path, thumb_name)
        if cached_img:
            return cached_img
        img = self.make_picture_thumb(picture_path)
        save_image_to_zip(zip_path, thumb_name, img, self.thumb_format)
        return img    

    
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
                 if file.lower().endswith(IMAGE_EXTENSIONS + VIDEO_EXTENSIONS + SOUND_EXTENSIONS + PDF_EXTENSIONS + TXT_EXTENSIONS + MD_EXTENSIONS)]
        if sort_by == 'NAME':
            sorted_files = sorted(files)
        elif sort_by == 'DATE':
            sorted_files = sorted(files, key=lambda x: os.path.getctime(os.path.join(self.directory, x)))
        else:
            sorted_files = files  
    
        return [os.path.join(self.directory, f) for f in sorted_files]
    

    def make_picture_thumb(self, picture_path):
        img = Image.open(picture_path)
        Lh = self.thumbSizeCache
        ratio = Lh / img.height        
        w, h = int(img.width * ratio), Lh
        img = img.resize((w, h), Image.LANCZOS)
        return (img)

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
        
    def make_pdf_thumb(self, file_path):
        img = self.get_grid_preview_from_pdf(file_path)
        Lh = self.thumbSizeCache
        ratio = Lh / img.height        
        w, h = int(img.width * ratio), Lh
        img = img.resize((w, h), Image.LANCZOS)
        return (img)


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
    
    
    def get_grid_preview_from_pdf(self, pdf_path, dpi=200, grid_size=3):
        pdf_document = fitz.open(pdf_path)
        page_count = pdf_document.page_count  
    
        if page_count < grid_size * grid_size:
            page = pdf_document.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)  
            pdf_document.close()
            return img
    
        n_tiles = grid_size * grid_size  
    
        indices = []
        for i in range(n_tiles):
            t = i / (n_tiles - 1)
            idx = int(round(t * (page_count - 1)))
            indices.append(idx)
    
        page_images = []
        for idx in indices:
            page = pdf_document.load_page(idx)  
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            page_images.append(img)
    
        pdf_document.close()
    
        tile_w, tile_h = page_images[0].size
        page_images = [im.resize((tile_w, tile_h), Image.LANCZOS) for im in page_images]
    
        grid_w = tile_w * grid_size
        grid_h = tile_h * grid_size
        grid_img = Image.new("RGB", (grid_w, grid_h))
    
        for i, im in enumerate(page_images):
            row = i // grid_size
            col = i % grid_size
            x = col * tile_w
            y = row * tile_h
            grid_img.paste(im, (x, y))
        return grid_img

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


                elif ext in TXT_EXTENSIONS:
                    img = self.txt_placeholder_img.copy()
                    ratio = min(Iw / img.width, Ih / img.height)
                    w, h = int(img.width * ratio), int(img.height * ratio)
                    img = img.resize((w, h), Image.LANCZOS)
                    photo_img = ImageTk.PhotoImage(img)
                    x = x + (max_img_width - w) // 2
                    y = y + (max_img_height - h) // 2
                    self.shadow(x, y, w, h)
                    img_id = self.canvas.create_image(x, y, anchor="nw", image=photo_img)
                    self.image_refs.append(photo_img)
                    self.canvas.tag_bind(img_id, "<Button-1>", lambda e, path=file_path: self.open_with_default_txt_viewer(path))
                    self.canvas.create_text(x+w//2, y+h//2, text="▶", fill="white", font=("Helvetica", max(20, w//6), "bold"))
                    self.canvas.create_text(x+w//2, y+h+20, text=f"{self.get_creation_date(file_path)}", fill="white", font=("Helvetica", 8, "bold"))

                elif ext in MD_EXTENSIONS:
                    img = self.md_placeholder_img.copy()
                    ratio = min(Iw / img.width, Ih / img.height)
                    w, h = int(img.width * ratio), int(img.height * ratio)
                    img = img.resize((w, h), Image.LANCZOS)
                    photo_img = ImageTk.PhotoImage(img)
                    x = x + (max_img_width - w) // 2
                    y = y + (max_img_height - h) // 2
                    self.shadow(x, y, w, h)
                    img_id = self.canvas.create_image(x, y, anchor="nw", image=photo_img)
                    self.image_refs.append(photo_img)
                    self.canvas.tag_bind(img_id, "<Button-1>", lambda e, path=file_path: self.open_with_default_md_viewer(path))
                    self.canvas.create_text(x+w//2, y+h//2, text="▶", fill="white", font=("Helvetica", max(20, w//6), "bold"))
                    self.canvas.create_text(x+w//2, y+h+20, text=f"{self.get_creation_date(file_path)}", fill="white", font=("Helvetica", 8, "bold"))

                elif ext in PDF_EXTENSIONS:
                    #img = self.get_first_page_from_pdf(file_path)
                    #img = self.get_grid_preview_from_pdf(file_path)      
                    img = self.get_cached_or_generate_pdf_thumb(file_path)
                        
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
                    img = self.get_cached_or_generate_picture_thumb(file_path)
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
            play_video_with_seek_and_pause(video_path,self.qModeBackground)
            
    def open_with_default_pdf_player(self, pdf_path):
        if self.qModeSoftwareView:
            if sys.platform.startswith('darwin'):
                subprocess.Popen(['open', pdf_path])
            elif os.name == 'nt':
                os.startfile(pdf_path)
            elif os.name == 'posix':
                subprocess.Popen(['xdg-open', pdf_path])
        else :
            view_pdf_zoom(pdf_path)

    def open_with_default_image_viewer(self, image_path):
        if self.qModeSoftwareView:
            if sys.platform.startswith('darwin'):
                subprocess.Popen(['open', image_path])
            elif os.name == 'nt':
                os.startfile(image_path)
            elif os.name == 'posix':
                subprocess.Popen(['xdg-open', image_path])        
        else :
            view_picture_zoom(image_path,self.qModeBackground)

    def open_with_default_audio_player(self, audio_path):
        if self.qModeSoftwareView:
            if sys.platform.startswith('darwin'):
                subprocess.Popen(['open', audio_path])
            elif os.name == 'nt':
                os.startfile(audio_path)
            elif os.name == 'posix':
                subprocess.Popen(['xdg-open', audio_path])
        else :
            play_audio_with_seek_and_waveform(audio_path)
            
    def open_with_default_txt_viewer(self, txt_path):
        if self.qModeSoftwareView:
            if sys.platform.startswith('darwin'):
                subprocess.Popen(['open', txt_path])
            elif os.name == 'nt':
                os.startfile(txt_path)
            elif os.name == 'posix':
                subprocess.Popen(['xdg-open', txt_path])
        else :
            #view_in_mode_opencv(txt_path)
            view_in_mode_txt(txt_path)
            
    def open_with_default_md_viewer(self, txt_path):
        if self.qModeSoftwareView:
            if sys.platform.startswith('darwin'):
                subprocess.Popen(['open', txt_path])
            elif os.name == 'nt':
                os.startfile(txt_path)
            elif os.name == 'posix':
                subprocess.Popen(['xdg-open', txt_path])
        else :
            #view_in_mode_opencv(txt_path)
            view_in_mode_txt(txt_path)

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
        
    def mode_Background_key(self, event): 
        self.qModeBackground = not self.qModeBackground

    def update_clock(self):
        self.clock_label.config(text=time.strftime('%H:%M:%S'))
        self.master.after(1000, self.update_clock)
        
    def exit_app_key(self, event): self.master.destroy()
    
    def on_left_click(self, event):
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        zone_width = 50
        zone_height = 50
        if event.x >= width - zone_width and event.y <= zone_height:
            self.master.destroy()
            # sys.exit(0)
    

    
    

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
    
    parser.add_argument('--ThumbSizeCache', type=int, default=540, help='ThumbSizeCache')
    
    parser.add_argument('--ThumbFormat', type=str, choices=["JPG", "PNG", "HEIF", "AVIF"], default="PNG",
                       help="Format for saved thumbnails (JPG, PNG, HEIF, AVIF)")
    parser.add_argument('--SortFiles', type=str, choices=["NAME", "DATE"], default="NAME",
                       help="Sort by NAME or DATE")
    args = parser.parse_args()
    root = tk.Tk()
    slideshow = Slideshow(root, args.Path, 
                          args.Cols, args.Rows, 
                          args.Mode, args.Workers, 
                          args.QCache, 
                          args.ThumbFormat, 
                          args.ThumbSizeCache, 
                          args.SortFiles, 
                          args.QModeSoftwareView)
    root.mainloop()
