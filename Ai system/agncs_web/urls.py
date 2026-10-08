from django.urls import path
from analyzer.views import home, analyze

urlpatterns = [path("", home, name="home"), path("analyze/", analyze, name="analyze")]
