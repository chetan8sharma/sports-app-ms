from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from .services.fly_run import FlyRunService
from .services.vertical_leap import VerticalLeapService
from .services.horizontal_jump import HorizontalJumpService
from .services.step_length import StepLengthService
from .services.lateral_shuffle import LateralShuffleService

ALLOWED_VIDEO_TYPES = [
    'video/mp4',
    'video/x-msvideo',
    'video/quicktime',
    'video/x-matroska',
]


class BaseVideoView(APIView):
    parser_classes = [MultiPartParser]
    service_class  = None

    def post(self, request):
        video_file = request.FILES.get('video')
        if not video_file:
            return Response(
                {'error': 'No video file provided. Send the file under the key "video".'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if video_file.content_type not in ALLOWED_VIDEO_TYPES:
            return Response(
                {
                    'error': f'Unsupported file type: {video_file.content_type}.',
                    'allowed_types': ['mp4', 'avi', 'mov', 'mkv'],
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            service = self.service_class()
            result  = service.process(video_file, video_file.name, **self.get_service_kwargs(request))
            return Response(result, status=status.HTTP_200_OK)
        except ValueError as e:
            return Response({'error': str(e)}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        except Exception as e:
            return Response(
                {'error': f'An unexpected error occurred: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def get_service_kwargs(self, request):
        """Override in subclasses to pass extra validated params to service.process()."""
        return {}


class FlyRunView(BaseVideoView):
    service_class = FlyRunService


class VerticalLeapView(BaseVideoView):
    service_class = VerticalLeapService

    def get_service_kwargs(self, request):
        raw = request.data.get('height_cm')
        if raw is None:
            raise ValueError("height_cm is required. Send it as a form field alongside the video.")
        try:
            height_cm = float(raw)
        except (TypeError, ValueError):
            raise ValueError("height_cm must be a number (e.g. 172 or 172.5).")
        if not (50 <= height_cm <= 250):
            raise ValueError("height_cm must be between 50 and 250 cm.")
        return {'person_height_cm': height_cm}

class HorizontalJumpView(BaseVideoView):
    service_class = HorizontalJumpService
 
class StepLengthView(BaseVideoView):
    service_class = StepLengthService

class LateralShuffleView(BaseVideoView):
    service_class = LateralShuffleService