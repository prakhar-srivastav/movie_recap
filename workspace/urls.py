from django.urls import path
from .audio_audit_app import audio_audit, get_audio_file, test_timestamp_recorder, improve, save_file
from .speech_maker import speech_maker, save_instance, regenerate, create_clip, create_yt_clip, combine_audio_instance, delete_selection, add_text_to_clip, fetch, fetch_image, save_text
from .video_audit_app import video_audit
from .views import home
from .workspace import workspace, control_panel, generate_free_flow_video
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [

    #Home
    path('',home, name = 'home'),
    #content_generations
    #Audio Audits
    path('audio_audit/', audio_audit, name = 'audio_audit'),
    path('audio_audit/api/get-audio/', get_audio_file, name='get_audio_file'),
    path('audio_audit/api/test-timestamp/', test_timestamp_recorder, name='test_timestamp'),
    path('audio_audit/api/improve/', improve, name = 'improve'),
    path('audio_audit/api/save_file/', save_file, name = 'save_file'),
    #Speech Maker
    path('speech_maker/', speech_maker, name = 'speech_maker'),
    path('speech_maker/api/save_instance/', save_instance, name = 'save_file'),
    path('speech_maker/api/regenerate/', regenerate, name = 'regenerate'),
    path('speech_maker/api/fetch/', fetch, name = 'fetch'),
    path('speech_maker/api/fetch_image/', fetch_image, name = 'fetch_image'),
    path('speech_maker/api/create_clip/', create_clip, name = 'create_clip'),
    path('speech_maker/api/create_yt_clip/', create_yt_clip, name = 'create_yt_clip'),
    path('speech_maker/api/save_text/', save_text, name = 'save_text'),
    path('speech_maker/api/combine_audio_instance/', combine_audio_instance, name = 'combine_audio_instance'),
    path('speech_maker/api/delete_selection/', delete_selection, name = 'delete_selection'),
    path('speech_maker/api/add_text_to_clip/', add_text_to_clip, name = 'add_text_to_clip'),
    #Video Audits
    path('video_audit/', video_audit, name = 'video_audit'),
    #workspace
    path('workspace/', workspace, name = 'workspace'),
    path('workspace/control_panel/', control_panel, name='get_workspace_info'),
    ] +  static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)