import gradio as gr
import os
from PIL import Image
import time
import threading

# 특정 폴더 경로 설정
image_folder = 'ui/image/CE~00_00_28'

# 폴더 내 모든 JPG 파일 리스트 불러오기
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 이미지 순차적으로 읽어오기
def get_next_image():
    while True:
        for image_path in image_files:
            yield Image.open(image_path)

# 슬라이드쇼 함수
def slideshow(img_display):
    gen = get_next_image()
    while True:
        img = next(gen)
        img_display.update(img)
        time.sleep(1)

# Gradio 인터페이스 설정
with gr.Blocks() as demo:
    img_display = gr.Image()
    
    def start_slideshow():
        thread = threading.Thread(target=slideshow, args=(img_display,))
        thread.start()
    
    gr.Button("Start Slideshow").click(fn=start_slideshow, inputs=None, outputs=None)

# 인터페이스 실행
demo.launch()
