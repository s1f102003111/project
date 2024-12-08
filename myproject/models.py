from django.db import models

class Track(models.Model):
    title = models.CharField(max_length=255)
    artist = models.CharField(max_length=255)
    lyrics = models.TextField()
    sentiment = models.JSONField()  # {'Joy': 0.1, 'Sadness': 0.2, ...}
    topics = models.JSONField()  # {'Topic1': 0.5, 'Topic2': 0.3, ...}

class Album(models.Model):
    title = models.CharField(max_length=255)
    artist = models.CharField(max_length=255)
    tracks = models.ManyToManyField(Track)

