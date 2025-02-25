"""
URL configuration for emotionsDetection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from player.views import music_player, music_home, music_about, detect_emotion

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", music_home, name="home"),
    path("about", music_about, name="about"),
    path("music-player", music_player, name="music-player"),
    path("detect-emotion", detect_emotion, name="emDetect")
]
