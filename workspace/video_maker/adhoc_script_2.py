from moviepy.editor import VideoFileClip
import moviepy.video.fx.all  as vfx
import os
from PIL import Image, ImageFile
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time
import psutil
from transformers import CLIPProcessor, CLIPModel
from pytube import YouTube
import moviepy.audio.fx.all as afx
from youtube_transcript_api import YouTubeTranscriptApi
import sys
import json
import traceback
import signal
import uuid
import shutil
import json
import subprocess
from moviepy.editor import TextClip, AudioClip, VideoClip, AudioFileClip, VideoFileClip, CompositeAudioClip, ImageSequenceClip, ImageClip, CompositeVideoClip
from moviepy.editor import concatenate_audioclips, concatenate_videoclips

def read_json(filename):
    try:
        j = open(filename, 'r')
        json_object =  json.loads(j.read())
        j.close()
        return json_object
    except:
        return dict()


def save_json(request_body, filename):
    json_object = json.dumps(request_body, indent=4)    
    with open(filename, "w") as outfile:
        outfile.write(json_object)


def warmth(clip, strength=0.8):
    def process_frame(get_frame, t):
        frame = get_frame(t)
        frame[:, :, 0] = frame[:, :, 0] * strength
        
        return frame

    return clip.transform(process_frame)


def invert(clip):
    import numpy as np
    def mirror_frame(get_frame, t):
        frame = get_frame(t)
        return np.fliplr(frame)

    return clip.transform(mirror_frame)


def create_clip(full_movie, yt_movie, timestamp, folder_path_a, folder_path_b, audio_clip, ret_path):
    start_time = 0
    per_image_time = 2
    clips = []
    video_clip = VideoFileClip(full_movie)
    video_clip.audio = None
    for key, value in timestamp.items():
        c_image = os.path.join(folder_path_a,str(value) + '.png')
        if int(key) % 2 == 1:
            continue
        clips.append(video_clip.subclip(value,value+per_image_time))

    final_clip = concatenate_videoclips(clips)
    final_clip.audio = audio_clip
    final_clip = vfx.lum_contrast(final_clip, lum = -4, contrast = 0.2)
    final_clip = warmth(final_clip)
    final_clip = invert(final_clip)

    final_clip.write_videofile(ret_path,
                        fps = 24,
                        threads = 16,
                        preset='ultrafast')


def generate_dict_1(system, content, mandatory_keys, model_type = 'gpt-4-turbo-preview', retries = 3):
        from openai import OpenAI
        client = OpenAI(api_key = 'sk-proj-L8a41e1mHEpmUjFK4wc0T3BlbkFJ7v9ugFqkIL01Hatfst3T')
        print('Fetching data from gpt api for system', system)
        for i in range(retries):
            print('try count: {},'.format(i+1))
            try:
                response = client.chat.completions.create(
                                model= model_type,
                                messages=[
                                    {
                                    "role": "system",
                                    "content": system
                                    },
                                    {
                                    "role": "user",
                                    "content": content
                                    },],
                                temperature=1,
                                max_tokens=4095,
                                top_p=1,
                                frequency_penalty=0,
                                presence_penalty=0
                                )
                
                message = response.dict()['choices'][0]['message']['content']
                message = message.replace('\n',' ')
                message = json.loads(message)
                for v in mandatory_keys:
                    assert v in message
                print('Done Fetching')
                return message
            except:
                print(traceback.format_exc())
                print(message)
                pass
        raise ValueError('Fetch Failed')



def get_captions(link):
    video_id = link.split("v=")[1]
    transcripts = YouTubeTranscriptApi.get_transcript(video_id)
    subtitle_text = " ".join([entry["text"] for entry in transcripts])
    # save_jsonfile({'content':subtitle_text}, output)
    return transcripts


from pydub import AudioSegment
from pydub.playback import play


def increase_speed(file_path, factor, time_to_contraint):
    audio = AudioSegment.from_file(file_path)
    if time_to_contraint is not None:
        factor = len(audio) / time_to_contraint 
    faster_audio = audio.speedup(playback_speed=factor)
    faster_audio.export(file_path, format="wav")


def extract_audio_from_chat_gpt(text, file_path, speaker_wav = 'alloy',speed_multiply = 1.2, time_to_contraint = None):
    from pathlib import Path
    from openai import OpenAI
    client = OpenAI(api_key = 'sk-proj-L8a41e1mHEpmUjFK4wc0T3BlbkFJ7v9ugFqkIL01Hatfst3T')

    response = client.audio.speech.create(
    input=text,
    model="tts-1-hd",
    voice=speaker_wav,
    )
    response.stream_to_file(file_path)
    increase_speed(file_path,speed_multiply,time_to_contraint)


def add_background_music(audio_file, result_file, folder , background_data = None, factor = 0.1, original_factor = 4.0, credit_file = None):
    
    music_data = []
    if background_data:
        background_data = set(background_data)
    for data in os.listdir(folder):
        if background_data is not None and data not in background_data:
            continue
        cur = os.path.join(folder, data)
        cur_music_data = {}
        cur_music_data['info'] = read_json(os.path.join(cur,'credits.json'))
        cur_music_data['path'] = os.path.join(cur,'background.wav')
        music_data.append(cur_music_data)
        

    from moviepy.editor import AudioFileClip, CompositeAudioClip
    if type(audio_file) == str:
        audio_music = AudioFileClip(audio_file)
    else:
        audio_music = audio_file
    audio_music = audio_music.fx(afx.multiply_volume, factor = original_factor)
    composite_list = [audio_music]
    rem = audio_music.duration
    start = 0
    credits = list()
    while True:
        if rem < 20:
            break            
        
        print("start", start)
        print("rem", rem)
        
        cur_music = random.choice(music_data)
        cur_music_o = AudioFileClip(cur_music['path'])
        cur_music_c = cur_music['info']
        duration_a = rem
        duration_b = cur_music_o.duration
        print("duration_a", duration_a)
        print("duration_b", duration_b)
        if duration_b < duration_a:
            cur_music_o = cur_music_o.set_start(start)
            cur_music_o = cur_music_o.fx(afx.multiply_volume, factor = factor)
            composite_list.append(cur_music_o)
            rem -= duration_b
            start += duration_b
            credits.append(cur_music_c)
        else:
            cur_music_o = cur_music_o.set_duration(rem+10).set_start(start).audio_fadeout(6)
            cur_music_o = cur_music_o.fx(afx.multiply_volume, factor = factor)
            composite_list.append(cur_music_o)
            credits.append(cur_music_c)
            rem -= duration_a
            start += duration_a

    
    composite_audio = CompositeAudioClip(composite_list)
    composite_audio.write_audiofile(result_file, codec='libmp3lame', fps = 44100)




full_movie = '/home/prakharrrr4/wolfy_ui/wolf/media/workspace/danish_girl/movie/movie_new_mutant.mp4'
yt_movie = '/home/prakharrrr4/wolfy_ui/wolf/workspace/video_maker/yt_movie.mp4'
timestamp = read_json('time.json')
folder_path_a = '/home/prakharrrr4/wolfy_ui/wolf/workspace/video_maker/image_of_movies/'
folder_path_b = '/home/prakharrrr4/wolfy_ui/wolf/workspace/video_maker/image_of_movies_1/images'
ret_path = 'test_result.mp4'
yt_link = 'https://www.youtube.com/watch?v=hlQKEhhiAj0'
system = '''User will give you are paragraph. This GPT job is to transform it giving the tone of human. Example of human tone (DONT USE ANY FACTUAL INFORMATION FROM THE EXAMPLE.  JUST USE THE SPEAKING STYLE). Keep the sentences short and precise. Don't elongate the answer. Keep number of words same
```
In mid-1920s Copenhagen, portrait artist Gerda Wegener asks her husband, a popular landscape artist and closeted trans woman (then going by the name Einar Wegener), to stand in for a female model, who is late arriving at their flat to pose for a painting Gerda is working on.
```

KEEP IT SIMPLE
DONT ACT SMART

DONT START WITH 'In mid-1920s Copenhagen,'
DONT USE ANY COMPLEX ENGLISH
WRITE IN ESSAY FORMAT LIKE A 8th STANDARD CHILD WRITING

Provide the final output in the following python dict. It should have a fields. Keep it within 200 words
            { "data
" : ""
            }'''

# create_clip(full_movie, yt_movie, timestamp, folder_path_a, folder_path_b, ret_path)
cap = get_captions(yt_link)
sentence_per_rank = 5
context = []
current_context = {'text' : '', 'start' : 99999999, 'end' : -1}
sen_count = 0

for snap in cap:
    text = snap['text'].strip()
    current_context['text'] += ' ' + text
    # start = 
    current_context['start'] = min(current_context['start'], snap['start'])
    current_context['end'] = max(current_context['end'], snap['duration'] + snap['start'])
    if text.endswith('.'):
        sen_count += 1
    
    if sen_count == sentence_per_rank:
        context.append(current_context)
        current_context = {'text' : '', 'start' : 99999999, 'end' : -1}
        sen_count = 0

audio_clips = []
itr = 0
for entry in [context]:
    text = entry['text']
    start = entry['start']
    end = entry['end']
    gpt_data = generate_dict_1(system,text,['data'])
    print(gpt_data['data'])
    hash_file = uuid.uuid4().hex + '.wav'
    extract_audio_from_chat_gpt(gpt_data['data'], hash_file, time_to_contraint = (end-start)*1000)
    audio_file = AudioFileClip(hash_file).set_start(start)
    original_factor = 4.0
    audio_file = audio_file.fx(afx.multiply_volume, factor = original_factor)
    audio_clips.append(audio_file)
    itr+=1


composite_audio = CompositeAudioClip(audio_clips)
create_clip(full_movie, yt_movie, timestamp, folder_path_a, folder_path_b, composite_audio, ret_path)