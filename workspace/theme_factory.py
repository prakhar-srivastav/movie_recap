from django.conf import settings
import os 
import sys




class ThemeFactory(object):

    def __init__(self, theme):
        self.theme_function = self.get_theme_function(theme)
        PATH_TO_THEMES =  os.path.join(settings.MEDIA_ROOT,'data/theme')
        if PATH_TO_THEMES not in sys.path:
            sys.path.append(PATH_TO_THEMES)

    def apply_theme(self,text):
        return self.theme_function(text)

    def get_theme_function(self, theme):
        module = __import__(theme)
        return getattr(module,'apply')


class VideoThemeFactory(object):

    def __init__(self, theme):
        self.theme_function = self.get_theme_function(theme)
        PATH_TO_THEMES =  os.path.join(settings.MEDIA_ROOT,'data/video_theme')
        if PATH_TO_THEMES not in sys.path:
            sys.path.append(PATH_TO_THEMES)
    
    def apply_theme(self,text):
        return self.theme_function(text)

    def get_theme_function(self, theme):
        module = __import__(theme)
        return getattr(module,'apply')



class AffectsThemeFactory(object):

    def __init__(self, theme):
        self.theme_function = self.get_theme_function(theme)
        PATH_TO_THEMES =  os.path.join(settings.MEDIA_ROOT,'data/affects_theme')
        if PATH_TO_THEMES not in sys.path:
            sys.path.append(PATH_TO_THEMES)
    
    def apply_theme(self,text):
        return self.theme_function(text)

    def get_theme_function(self, theme):
        module = __import__(theme)
        return getattr(module,'apply')


class CaptionThemeFactory(object):

    def __init__(self, theme):
        self.theme_function = self.get_theme_function(theme)
        PATH_TO_THEMES =  os.path.join(settings.MEDIA_ROOT,'data/caption_theme')
        if PATH_TO_THEMES not in sys.path:
            sys.path.append(PATH_TO_THEMES)
    
    def apply_theme(self,text):
        return self.theme_function(text)

    def get_theme_function(self, theme):
        module = __import__(theme)
        return getattr(module,'apply')
