from django.urls import path
from .views import audio_audit, get_audio_file, test_timestamp_recorder, improve, save_file
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', audio_audit, name = 'audio_audit'),
    path('api/get-audio/', get_audio_file, name='get_audio_file'),
    path('api/test-timestamp/', test_timestamp_recorder, name='test_timestamp'),
    path('api/improve/', improve, name = 'improve'),
    path('api/save_file/', save_file, name = 'save_file')
    ] +  static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)