# Slideshow Panel Media

[![Version](https://img.shields.io/badge/version-2.1-green.svg)](https://github.com/lemoinep/MiniSlideshowPanelMediaPlayer)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

---

## Objective

The program is an advanced multimedia viewer written in Python that can display and browse images, videos, PDFs, text files, and Markdown documents from folders or ZIP archives.

It uses Tkinter for the main interface, OpenCV and Pillow for image processing and video playback, PyMuPDF for PDF rendering, pygame/pydub for audio, and markdown-it to render Markdown as images. 

The code provides many image-processing features (sharpening, adaptive contrast, CLAHE, retinex, pointillism/oil-painting effects, edge detection, saliency, stereo anaglyphs, region removal, automatic cropping, etc.) that can be applied in full-screen view with interactive zoom.

For videos, it manages a cache system (folder or ZIP) with a JSON configuration to store precomputed thumbnails/frames, which speeds up media navigation.

The program also offers dedicated viewers:  
- a full-screen “Picture Zoom” mode with mouse-wheel zoom, screenshot capture, and multiple filters controlled by keyboard shortcuts;  
- a “PDF Viewer” mode with high-resolution rendering, zoom, page navigation (left/right click areas, space, 1/3 keys), a visual progress bar, and text enhancement options;  
- an “MD Viewer” mode that converts Markdown into an OpenCV image with support for headings, lists, code blocks, and links, then displays it with zoom.


## 📝 **Author**

**Dr. Patrick Lemoine**  
*Engineer Expert in Scientific Computing*  
[LinkedIn](https://www.linkedin.com/in/patrick-lemoine-7ba11b72/)

---

