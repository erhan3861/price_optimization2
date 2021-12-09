"""PricePredictionAPI URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
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
from django.urls import path,include
from api import views
#from django.conf.urls import url, include, path
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf.urls.static import static



urlpatterns = [
    path('admin/', admin.site.urls),
    #path('v1/api/', include('api.urls')),
    #path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    # add these to configure our home page (default view) and result web page
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('MLfunction/', views.ML_function, name='MLfunction'),
    path('result/', views.result, name='result'),
    path('data/', views.get_data, name='get_data'),
    path('getfile/', views.getfile, name='getfile'),
    path('set_store/', views.set_store, name='set_store'),
    path('help/', views.help, name='help'),

    
] 

urlpatterns += staticfiles_urlpatterns()