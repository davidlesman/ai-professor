from django.urls import path

from .views import IndexPage, Send


urlpatterns = [
    path("", IndexPage.as_view(), name="index"),
    path("send/", Send.as_view()),
]
