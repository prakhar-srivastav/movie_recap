

from pydub import AudioSegment
from pydub.playback import play
import time

def increase_speed(file_path, factor, time_to_contraint):
    audio = AudioSegment.from_file(file_path)
    if time_to_contraint is not None:
        factor = len(audio) / time_to_contraint 
    if factor <= 1.0:
        return
    faster_audio = audio.speedup(playback_speed=factor)
    faster_audio.export(file_path, format="wav")


def extract_audio_from_chat_gpt(text, file_path, speaker_wav = 'nova',speed_multiply = 1.0, time_to_contraint = None):
    from pathlib import Path
    from openai import OpenAI
    client = OpenAI(api_key = 'sk-proj-L8a41e1mHEpmUjFK4wc0T3BlbkFJ7v9ugFqkIL01Hatfst3T')

    response = client.audio.speech.create(
    input=text,
    model="tts-1-hd",
    voice=speaker_wav,
    )
    response.stream_to_file(file_path)
    # increase_speed(file_path,speed_multiply,time_to_contraint)
    time.sleep(20)
    # from moviepy.editor import AudioFileClip
    # audio_clip = AudioFileClip(file_path)
    # import moviepy.video.fx.all as vfx
    # audio_clip = audio_clip.fx(vfx.multiply_speed, factor=1.2)
    # audio_clip.write_audiofile(file_path)

