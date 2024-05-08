from pydub import AudioSegment
import numpy as np
import librosa
from scipy.signal import fftconvolve
import soundfile as sf
import librosa, os
import uuid
from . import defaults
import json
import shutil
from moviepy.editor import AudioClip, VideoClip, AudioFileClip, VideoFileClip, CompositeAudioClip, CompositeVideoClip, ImageSequenceClip, ImageClip
from moviepy.editor import concatenate_audioclips, concatenate_videoclips
import moviepy.video.fx.all as vfx 
import random
import traceback
import moviepy.audio.fx.all as afx


def get_hash():
    return defaults.temp_path +  uuid.uuid4().hex


"""
A U D I O
A F F E C T S
"""

def resample_audio(input_path, output_path, target_sr):
    audio, sr = librosa.load(input_path, sr=None)
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    sf.write(output_path, audio_resampled, target_sr)


def apply_exponential_decay(ir, decay_rate=0.0001):
    """
    Applies an exponential decay to the impulse response.
    decay_rate controls the rate of the decay.
    """
    n = np.arange(len(ir))
    decay_curve = np.exp(-decay_rate * n)
    return ir * decay_curve


def add_reverb(audio_path):
    resample_audio(audio_path, audio_path, 24000)
    audio, sr_audio = librosa.load(audio_path, sr=None, mono=True)
    ir, sr_ir = librosa.load(defaults.ir_path, sr=None, mono=True)
    ir = apply_exponential_decay(ir)
    reverberated_audio = fftconvolve(audio, ir, mode='full')
    reverberated_audio /= np.max(np.abs(reverberated_audio))
    sf.write(audio_path, reverberated_audio, sr_audio)


def make_silence(duration):
    return AudioClip(lambda t: 0, duration = duration)


def combine_audio(audio_files, ret_path):
    audio_clips = []
    for audio_file in audio_files:
        audio_clip = AudioFileClip(audio_file)
        audio_clips.append(audio_clip)
    final_clip = concatenate_audioclips(audio_clips)    
    final_clip.write_audiofile(ret_path)



"""
V I D E O 
A F F E C T S
"""


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


def shaky(clip, amplitude=0.2):

    def fl(gf, t):
        dx = float(np.random.randint(-amplitude*1000, amplitude*1000)) / 1000
        dy = float(np.random.randint(-amplitude*1000, amplitude*1000)) / 1000
        return gf(t).clip(dx, dy)
    
    return clip.fl(fl)


def overlay(video_file, duration, opacity = 0.2, fadeout_duration = 3):
    composite_list = []
    rem = duration
    start = 0

    while rem > 0:
        clip = VideoFileClip(video_file)        
        duration_a = rem
        duration_b = clip.duration

        if clip.duration < rem:
            # clip = vfx.fadeout(clip.set_start(start), fadeout_duration)
            clip = clip.set_start(start)
            composite_list.append(clip)
            rem -= clip.duration
            start += clip.duration
        else:
            clip = clip.set_duration(rem).set_start(start)
            composite_list.append(clip)
            start += rem
            rem = 0

    result = CompositeVideoClip(composite_list).with_opacity(opacity)
    return result

"""
A P I
"""


def generate_dict_1(system, content, mandatory_keys, model_type = 'gpt-4-turbo-preview', retries = 3):
        from openai import OpenAI
        client = OpenAI(api_key = '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')
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


def generate_image_s(
    prompt,
    negative_prompt,
    result_path,
    image_type='photographic',
    number_of_images=1,
    positive_weight = 0.7,
    negative_weight = None,
    clip_guidance_preset = None,
    cfg_scale = 7
    ):
    import os
    import requests
    import base64

    api_host ='https://api.stability.ai'
    url = f"{api_host}/v1/engines/list"
    api_key = 'sk-RmjRShKlYdG3VArpSUYKmQCashOPQTaww3IqplKOiZ1razNV'
    MODEL = 'stable-diffusion-xl-1024-v1-0'

    text_prompts = [{'text' : prompt, 'weight' : positive_weight}]
    if negative_prompt: text_prompts.append({'text' : negative_prompt, 'weight' : negative_weight})

    response = requests.post(
        f"{api_host}/v1/generation/{MODEL}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": text_prompts
            ,
            "height": 1024,
            "width": 1024,
            "samples": number_of_images,
            "steps": 30,
        },
        )

    data = response.json()

    for i, image in enumerate(data['artifacts']):
        file = open(result_path,'wb')
        file.write(base64.b64decode(image["base64"]))
        file.close()
    


"""
F I L E
P A R S I N G
"""

def save_json(request_body, filename):
    json_object = json.dumps(request_body, indent=4)    
    with open(filename, "w") as outfile:
        outfile.write(json_object)


def read_json(filename):
    j = open(filename, 'r')
    json_object =  json.loads(j.read())
    j.close()
    return json_object


"""
B A C K G R O U N D
M U S I C
"""

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
            cur_music_o = cur_music_o.set_duration(rem).set_start(start).audio_fadeout(6)
            cur_music_o = cur_music_o.fx(afx.multiply_volume, factor = factor)
            composite_list.append(cur_music_o)
            credits.append(cur_music_c)
            rem -= duration_a
            start += duration_a

    
    composite_audio = CompositeAudioClip(composite_list)
    composite_audio.write_audiofile(result_file, codec='libmp3lame', fps = 44100)

    save_json(credits, credit_file)

