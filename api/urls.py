from django.urls import path
from .views import CompareCVsView

urlpatterns = [
    path('compare-cvs/', CompareCVsView.as_view(), name='compare-cvs'),
]