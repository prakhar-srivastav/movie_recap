from django.db import models


class Texts(models.Model):
    text = models.CharField(max_length=1000)