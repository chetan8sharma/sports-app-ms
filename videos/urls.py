from django.urls import path
from .views import FlyRunView, VerticalLeapView, HorizontalJumpView, StepLengthView, LateralShuffleView

urlpatterns = [
    path('fly-run/',        FlyRunView.as_view(),        name='fly-run'),
    path('vertical-leap/',  VerticalLeapView.as_view(),  name='vertical-leap'),
    path('horizontal-jump/', HorizontalJumpView.as_view(), name='horizontal-jump'),
    path('step-length/',    StepLengthView.as_view(),    name='step-length'),
    path('lateral-shuffle/', LateralShuffleView.as_view(), name='lateral-shuffle'),
]