from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django import forms
from django.http import JsonResponse
from django.templatetags.static import static
from django.conf import settings
from .audio_generator import AudioGenerator, ThemeFactory

import sys
import os
import json
import uuid
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

def get_youtube_captions(link):
    video_id = link.split("v=")[1]
    transcripts = YouTubeTranscriptApi.get_transcript(video_id)
    subtitle_text = " ".join([entry["text"] for entry in transcripts])
    return subtitle_text

def get_speaker_choices():
    PATH = os.path.join(settings.MEDIA_ROOT,'data/audio_data')
    speaker = [x.replace('.wav','') for x in os.listdir(PATH)]
    return speaker

def get_workspace_choices():
    workspace_path = os.path.join(settings.MEDIA_ROOT,'workspace')
    os.makedirs(workspace_path, exist_ok = True)
    return os.listdir(workspace_path)

def get_theme_choices():
    return ThemeFactory()._theme_

def get_speaker_path(speaker_name):
    PATH = os.path.join(settings.MEDIA_ROOT,'data/audio_data')
    return PATH + '/' + speaker_name + '.wav'
    
class AudioAuditForm(forms.Form):
    text_input = forms.CharField(label='Your text', max_length=1000)
    speakers = get_speaker_choices()
    choices = []
    for i in range(len(speakers)):
        choices.append((i+1,speakers[i]))
    speaker_selection = forms.ChoiceField(label='Choose a speaker', choices= choices)


@csrf_exempt
def get_audio_file(request):
    data = json.loads(request.body.decode('utf-8'))['data']
    youtube_link = data['youtube_link']
    theme = data['theme']
    transcript = data['transcript']
    speaker_name = data['speaker_name']
    text = ''
    if youtube_link is None or youtube_link.strip() == '':
        text = transcript
    else:
        text = get_youtube_captions(youtube_link)

    global audio_generator

    audio_generator = AudioGenerator()

    audio_generator.synthesize(text, get_speaker_path(speaker_name))
    audio_file_name = uuid.uuid4().hex + '.wav'
    audio_url = os.path.join(settings.MEDIA_ROOT,audio_file_name)
    web_accessible_url = settings.MEDIA_URL + audio_file_name
    audio_generator.save_file(audio_url)
    return JsonResponse({'audio_url' : web_accessible_url})

@csrf_exempt
def improve(request):
    data = json.loads(request.body.decode('utf-8'))['data']
    timestamps = data['timestamps']
    timestamps = [float(x) for x in timestamps]
    audio_generator.feedback(timestamps, is_seconds = True)
    audio_file_name = uuid.uuid4().hex + '.wav'
    audio_url = os.path.join(settings.MEDIA_ROOT,audio_file_name)
    web_accessible_url = settings.MEDIA_URL + audio_file_name
    audio_generator.save_file(audio_url)
    return JsonResponse({'audio_url' : web_accessible_url})

def audio_audit(request):
    speaker_list = get_speaker_choices()
    workspace_list = get_workspace_choices()
    theme_list = get_theme_choices()
    context = {
        'workspace_list' : workspace_list,
        'speaker_list' : speaker_list,
        'theme_list' : theme_list
    }
    return render(request, 'audio_audit/main.html', context)

@csrf_exempt
def save_file(request):
    data = json.loads(request.body.decode('utf-8'))['data']
    filename = data['filename']
    workspace_name = data['workspace']
    folder = os.path.join(settings.MEDIA_ROOT,'workspace/{}/audio'.format(workspace_name))
    os.makedirs(folder, exist_ok = True)
    audio_generator.save_file(os.path.join(folder, filename))
    return HttpResponse('true')

def test_timestamp_recorder(request):
    return render(request, 'audio_audit/timestamp_recorder.html')


# def quote_preset()