from django.urls import path
from . import views


urlpatterns = [
    path('', views.classifier, name='classifier'),
    path('part1', views.part1, name='part1'),
    path('part2', views.part2, name='part2'),
]

