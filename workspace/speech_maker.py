from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django import forms
from django.http import JsonResponse
from django.templatetags.static import static
from django.conf import settings

import sys
import os
import json
import uuid
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from .audio_generator import ThemeFactory

def get_speaker_path(speaker_name):
    PATH = os.path.join(settings.MEDIA_ROOT,'data/audio_data')
    return PATH + '/' + speaker_name + '.wav'


def filesystem_to_media_url(absolute_file_path):
    media_root = os.path.normpath(settings.MEDIA_ROOT)
    absolute_file_path = os.path.normpath(absolute_file_path)
    
    if absolute_file_path.startswith(media_root):
        relative_path = absolute_file_path[len(media_root):].lstrip(os.path.sep)
        media_url = os.path.join(settings.MEDIA_URL, relative_path).replace('\\', '/')
        return media_url
    else:
        raise ValueError("The provided file path is not under MEDIA_ROOT.")


@csrf_exempt
def speech_maker(request):
    data = request.GET
    youtube_link = data.get('youtube_link',"").strip()
    theme = data.get("theme","").strip()
    transcript = data.get("transcript","").strip()
    speaker_name = data.get("speaker_name","").strip()
    workspace = data.get("workspace","").strip()
    instance_name = data.get('instance_name', "").strip()
    
    theme_factory = ThemeFactory()

    global wolfy_tts_middleware
    from workspace.video_maker.preset_for_movie import create_yt_clip, create_clip

    if instance_name == "":
        text = transcript
        wolfy_tts_middleware = theme_factory.get_middleware(workspace, theme = theme)
        wolfy_tts_middleware.synthesize(text, speaker_name, youtube_link = youtube_link)
    else:
        wolfy_tts_middleware = theme_factory.get_middleware(workspace, theme = theme, _id_ = instance_name)

    content_context = wolfy_tts_middleware.get_context()
    movie_url = wolfy_tts_middleware.get_movie_url()
    movie_url = filesystem_to_media_url(movie_url)
    print("MOVIE_URL", movie_url)
    data = {
        'workspace' : workspace,
        'instance' : wolfy_tts_middleware.get_instance()
    }
    result_video_url = wolfy_tts_middleware.get_result_video_url()
    if os.path.exists(result_video_url):
        result_video_url = filesystem_to_media_url(result_video_url)
    else:
        create_clip(data)
        result_video_url = filesystem_to_media_url(result_video_url)
    print("RESULT_VIDEO_URL", result_video_url)

    result_yt_video_url = wolfy_tts_middleware.get_result_yt_video_url()
    if os.path.exists(result_yt_video_url):
        result_yt_video_url = filesystem_to_media_url(result_yt_video_url)
    else:
        create_yt_clip(data)
        result_yt_video_url = filesystem_to_media_url(result_yt_video_url)
    
    print("RESULT_YT_VIDEO_URL", result_yt_video_url)

    for key, value in content_context.items():
        clip_file = value['video_url_fs']
        if os.path.exists(clip_file):
            continue
        temp_data = data
        temp_data['_ids_'] = key
        create_clip(temp_data)
    
    return render(request,
                'speech_maker/main.html',
                context = {
                    'workspace' : workspace,
                    'content_context' : content_context,
                    'instance_name' : instance_name,
                    'movie_url' : movie_url,
                    'result_video_url' : result_video_url,
                    'result_yt_video_url' : result_yt_video_url
                })    
    
@csrf_exempt
def save_instance(request):
    data = json.loads(request.body.decode('utf-8'))['data']
    filename = data['filename']
    wolfy_tts_middleware.save_file(filename)
    return HttpResponse('true')


@csrf_exempt
def save_text(request):
    data = json.loads(request.body.decode('utf-8'))['data']    
    _ids_ = data['_ids_']
    workspace = data['workspace']
    instance_name = data['instance_name']
    text = data['text']
    theme_factory = ThemeFactory()
    wolfy_tts_middleware = theme_factory.get_middleware(workspace, _id_ = instance_name)
    wolfy_tts_middleware.save_text(_ids_, text)
    context = wolfy_tts_middleware.regenerate_by_ids(_ids_)
    return JsonResponse(context)



@csrf_exempt
def combine_audio_instance(request):
    data = json.loads(request.body.decode('utf-8'))['data']
    audio_list = data['audio_list']
    workspace = data['workspace']
    a = ThemeFactory().get_middleware(workspace, _id_ = audio_list[0])

    for i in range(1,len(audio_list)):
        if audio_list[i].startswith('s_'):
            silence = int(audio_list[i].split('_')[-1])
            a.add_silence_audio(silence)
        else:
            a += ThemeFactory().get_middleware(workspace, _id_ = audio_list[i])
    return HttpResponse('true')


@csrf_exempt
def regenerate(request):
    data = json.loads(request.body.decode('utf-8'))['data']    
    _ids_ = data['_ids_']
    workspace = data['workspace']
    instance_name = data['instance_name']
    theme_factory = ThemeFactory()
    wolfy_tts_middleware = theme_factory.get_middleware(workspace, _id_ = instance_name)

    context = wolfy_tts_middleware.regenerate_by_ids(_ids_)
    return JsonResponse(context)


@csrf_exempt
def fetch(request):
    data = json.loads(request.body.decode('utf-8'))['data']    
    _ids_ = data['_ids_']
    workspace = data['workspace']
    instance_name = data['instance_name']
    theme_factory = ThemeFactory()
    wolfy_tts_middleware = theme_factory.get_middleware(workspace, _id_ = instance_name)
    context = wolfy_tts_middleware.get_context()
    timestamp = context[_ids_]['all'].get('movie_timestamps',[])
    return JsonResponse({'timestamp' : timestamp})


@csrf_exempt
def fetch_image(request):
    data = json.loads(request.body.decode('utf-8'))['data']    
    _ids_ = data['_ids_']
    workspace = data['workspace']
    instance_name = data['instance_name']
    theme_factory = ThemeFactory()
    wolfy_tts_middleware = theme_factory.get_middleware(workspace, _id_ = instance_name)
    context = wolfy_tts_middleware.get_context()
    timestamp = context[_ids_]['all'].get('movie_timestamps',[])
    _main_folder_path_ = wolfy_tts_middleware.get_main_folder_path()
    _main_folder_path_ = os.path.join(_main_folder_path_,'image/movie/images')
    image_path = [os.path.join(_main_folder_path_,str(t)+'.png') for t in timestamp]
    image_path = [filesystem_to_media_url(path) for path in image_path]
    return JsonResponse({
                        'image_path' : image_path,
                        'timestamp' : timestamp })


@csrf_exempt
def create_clip(request):
    data = json.loads(request.body.decode('utf-8'))['data']
    from workspace.video_maker.preset_for_movie import create_clip
    source_url = create_clip(data)
    result = {}
    result['source_url'] = filesystem_to_media_url(source_url)
    return JsonResponse(result)


@csrf_exempt
def create_yt_clip(request):
    data = json.loads(request.body.decode('utf-8'))['data']
    from workspace.video_maker.preset_for_movie import create_yt_clip
    source_url = create_yt_clip(data)
    result = {}
    result['source_url'] = filesystem_to_media_url(source_url)
    return JsonResponse(result)


@csrf_exempt
def delete_selection(request):
    data = json.loads(request.body.decode('utf-8'))['data']
    _ids_ = data['_ids_']
    wolfy_tts_middleware.delete_by_ids(_ids_)
    return HttpResponse('true')


@csrf_exempt
def add_text_to_clip(request):
    data = json.loads(request.body.decode('utf-8'))['data']
    from workspace.video_maker.preset_for_movie import add_text_to_clip
    source_url = add_text_to_clip(data)
    result = {}
    result['source_url'] = filesystem_to_media_url(source_url)
    return JsonResponse(result)
