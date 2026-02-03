from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'hadiths', views.HadithViewSet, basename='hadith')
router.register(r'hadith-embeddings', views.HadithEmbeddingViewSet, basename='hadith-embedding')
router.register(r'hadith-sources', views.HadithSourceViewSet, basename='hadith-source')
router.register(r'hadith-chunk-embeddings', views.HadithChunkEmbeddingViewSet, basename='hadith-chunk-embedding')
router.register(r'siyer-sections', views.SiyerSectionViewSet, basename='siyer-section')
router.register(r'siyer-chunk-embeddings', views.SiyerChunkEmbeddingViewSet, basename='siyer-chunk-embedding')

urlpatterns = [
    path('', include(router.urls)),
]
