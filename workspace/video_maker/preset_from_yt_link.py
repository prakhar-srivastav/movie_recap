import cv2
from moviepy.editor import VideoFileClip
import numpy as np
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import sys
import json
from moviepy.editor import VideoFileClip
import os
import random
from PIL import Image, ImageFile


def download_yt(link):
    from pytube import YouTube
    yt = YouTube(link)
    stream = yt.streams.first()
    stream.download()   


def save_random_frame(video_path, image_path):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    random_timestamp = random.uniform(0, duration)
    clip.save_frame(image_path, t=random_timestamp)
    print(f"Saved frame at {random_timestamp:.2f} seconds to {image_path}")


def load_frames_as_pil(video_path, interval=0.5):

    clip = VideoFileClip(video_path)
    frames = []
    duration = clip.duration
    current_time = 0
    while current_time <= duration:
        frame = clip.get_frame(current_time)
        pil_image = Image.fromarray(frame)
        frames.append(pil_image)
        print(current_time)
        current_time += interval

    return frames


def find_frame_with_features(video_path, image_path):

    movie_images = load_frames_as_pil(video_path, interval = 0.5)

    import pdb; pdb.set_trace()

    print(len(images))

    text = _sentence_splitter_(full_text)
    text_map = {}
    for te in text:
        text_map[te] = []

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs.to("cuda"))
    logits_per_image = outputs.logits_per_image
    ma = 0
    _id_ = image0
    for te in text:
        for j in range(len(_id_)):
            yo = int(_id_[j].split('/')[-1].split('.')[0])
            text_map[te].append((yo, float(logits_per_image[j][ma])))
        ma += 1

    save_json(text_map, result_path)


full_movie = '/home/prakharrrr4/wolfy_ui/wolf/media/workspace/danish_girl/movie/movie_new_mutant.mp4'
yt_movie = '/home/prakharrrr4/wolfy_ui/wolf/workspace/video_maker/yt_movie.mp4'


# Usage
video_path = full_movie
from moviepy.editor import VideoFileClip
image_path = 'random_image.jpeg'
save_random_frame(video_path, image_path)
timestamp = find_frame_with_features(video_path, image_path)
print(f"Best matching frame is at timestamp: {timestamp}")
