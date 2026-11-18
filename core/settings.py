import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'change-me')
DEBUG = os.getenv('DEBUG', 'False') == 'True'
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'corsheaders',
    'django.contrib.contenttypes',
    'django.contrib.auth',          # ← DRF needs this internally
    'rest_framework',
    'videos',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'videos.middleware.JWTAuthMiddleware',
]

# Option A: Allow everyone (Good for public APIs/Testing)
CORS_ALLOW_ALL_ORIGINS = True 

# Option B: Allow specific apps (Best for Production)
# CORS_ALLOWED_ORIGINS = [
#     "http://localhost:3000",
#     "https://your-frontend-app.com",
# ]

# Tell DRF to use NO authentication or permissions by default
# We handle auth ourselves via the JWT middleware
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [],
    'DEFAULT_PERMISSION_CLASSES': [],
}

ROOT_URLCONF = 'core.urls'
WSGI_APPLICATION = 'core.wsgi.application'

# JWT — must match your Go backend secret
# JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
# JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')

JWT_SECRET_KEY = os.getenv('JWT_SECRET')
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')

# Allow large video uploads (500 MB)
DATA_UPLOAD_MAX_MEMORY_SIZE = 524288000
FILE_UPLOAD_MAX_MEMORY_SIZE = 524288000

STATIC_URL = '/static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'