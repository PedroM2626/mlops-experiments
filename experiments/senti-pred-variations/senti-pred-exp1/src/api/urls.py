"""
Configuração de URLs para a API Django do projeto Senti-Pred.
"""
from django.urls import path
from .views import SentimentPredictionView, ModelInfoView, health_check

urlpatterns = [
    path('predict/', SentimentPredictionView.as_view(), name='predict'),
    path('model-info/', ModelInfoView.as_view(), name='model-info'),
    path('health/', health_check, name='health'),
]