from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django import forms
from django.http import JsonResponse
from django.templatetags.static import static
from django.conf import settings
from .audio_generator import ThemeFactory

import sys
import os
import json
import uuid


def get_video_theme_choices():
    PATH = os.path.join(settings.MEDIA_ROOT,'data/video_theme')
    themes = [x.replace('.py','') for x in os.listdir(PATH)]
    return themes

def get_workspace_choices():
    workspace_path = os.path.join(settings.MEDIA_ROOT,'workspace')
    os.makedirs(workspace_path, exist_ok = True)
    return os.listdir(workspace_path)

def get_caption_list():
    PATH = os.path.join(settings.MEDIA_ROOT,'data/caption_theme')
    themes = [x.replace('.py','') for x in os.listdir(PATH)]
    return themes

def get_background_music_list():
    PATH = os.path.join(settings.MEDIA_ROOT,'data/background_music')
    return os.listdir(PATH)

def get_affects_list():
    PATH = os.path.join(settings.MEDIA_ROOT,'data/affects_theme')
    themes = [x.replace('.py','') for x in os.listdir(PATH)]
    return themes

def filesystem_to_media_url(absolute_file_path):
    media_root = os.path.normpath(settings.MEDIA_ROOT)
    absolute_file_path = os.path.normpath(absolute_file_path)
    
    if absolute_file_path.startswith(media_root):
        relative_path = absolute_file_path[len(media_root):].lstrip(os.path.sep)
        media_url = os.path.join(settings.MEDIA_URL, relative_path).replace('\\', '/')
        return media_url
    else:
        raise ValueError("The provided file path is not under MEDIA_ROOT.")

def get_files_ending_with_suffix(workspace, suffix):
    workspace_path = os.path.join(settings.MEDIA_ROOT,'workspace')
    workspace_path = os.path.join(workspace_path, workspace) 
    os.makedirs(workspace_path, exist_ok = True)

    files = []
    for file in os.listdir(workspace_path):
        ok = False
        for suf in suffix:
            if file.endswith(suf):
                ok = True
        if ok:
            full_file_path = os.path.join(workspace_path,file)
            files.append((file.split('.')[0], filesystem_to_media_url(full_file_path)))
    return files

def get_audio_files_from_workspace(workspace):
    return get_files_ending_with_suffix(os.path.join(workspace,'audio'), ['wav'])

def get_video_files_from_workspace(workspace):
    return get_files_ending_with_suffix(os.path.join(workspace,'video'), ['mp3'])

def get_audio_context_from_workspace(workspace):
    audio_folder = os.path.join(settings.MEDIA_ROOT,'workspace/{}/audio'.format(workspace))
    audio_context = {}
    for _id_ in  os.listdir(audio_folder):
        instance = ThemeFactory().get_middleware(workspace, _id_ = _id_)
        audio_context[_id_] = {
                                'time' : round(instance.get_total_time()/(1000*60),2),
                                'name' : _id_
                            }
    return audio_context

@csrf_exempt
def reload_audio_context(request):
    data = json.loads(request.body.decode('utf-8'))['data']
    workspace = data['workspace']
    audio_context = get_audio_context_from_workspace(workspace)
    return JsonResponse(audio_context)


@csrf_exempt
def control_panel(request):
    data = request.GET
    workspace = data['workspace']
    audio_context = get_audio_context_from_workspace(workspace)
    # video_files = get_video_files_from_workspace(workspace)
    # video_theme_files = get_video_theme_choices()
    # caption_list = get_caption_list()
    # affects_list = get_affects_list()
    background_music_list = get_background_music_list()
    context = {
        'workspace' : workspace,
        'audio_context' : audio_context,
        # 'video_list' : video_files,
        # 'video_theme' : video_theme_files,
        # 'caption_list' : caption_list,
        # 'affects_list' : affects_list,
        # 'background_music_list' : background_music_list,
            }
    return render(request, 'workspace/control_panel/display_workspace.html', context)


@csrf_exempt
def generate_free_flow_video(request):
    data = json.loads(request.body.decode('utf-8'))['data']
    workspace = data.get('workspace')
    audio_selection = data.get('audio_selection')
    theme_type = data.get('theme_type')
    affects = data.get('affects')
    caption_type = data.get('caption_type')
    background_music_selections = data.get('background_music_selections')

    """
    step 1: Make a video from the theme type
    step 2: Add affect in cascade order. Keep it simple
    step 3: Add captions 
    step 4: Add Background Music
    step 5: Prepare and send descriptions
    """
    return HttpResponse('true')

def workspace(request):
    context = {
        'workspace_list' : get_workspace_choices()
    }
    return render(request, 'workspace/main.html', context)