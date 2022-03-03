from django.shortcuts import render

# Create your views here.
from django.urls import path

#from api import views
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('api.urls')),
]
"""
urlpatterns = [
    path('price/', PricePrediction.as_view(), name = 'price_prediction'),
]
"""