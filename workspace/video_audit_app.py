from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django import forms
from django.http import JsonResponse
from django.templatetags.static import static
from django.conf import settings
from .theme_factory import VideoThemeFactory
from .workspace import get_files_ending_with_suffix

import sys
import os
import json
import uuid

def get_video_theme_choices():
    PATH = os.path.join(settings.MEDIA_ROOT,'data/video_theme')
    themes = [x.replace('.py','') for x in os.listdir(PATH)]
    return themes

def get_audio_list(workspace):
    PATH = os.path.join(settings.MEDIA_ROOT,'data/workspace/' + workspace)
    return get_files_ending_with_suffix(PATH, ['wav'])

@csrf_exempt
def video_audit(request):
    data = json.loads(request.body.decode('utf-8'))['data']
    workspace = data['workspace']
    video_type_list = get_video_type_list()
    audio_list = get_audio_file_list(workspace)
    context = {
        
        'audio_list' : audio_list,
    }
    return render(request, 'video_audit/main.html', context)

