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

def combine_audio_files(request):
    return HttpResponse('True')