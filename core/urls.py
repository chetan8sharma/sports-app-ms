from django.urls import path, include

urlpatterns = [
    path('api/videos/', include('videos.urls')),
]