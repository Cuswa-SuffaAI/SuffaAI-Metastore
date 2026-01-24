from rest_framework import serializers
from .models import Hadith, HadithEmbedding, HadithSource, HadithChunkEmbedding


class HadithSourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = HadithSource
        fields = ['id', 'name', 'reference']


class HadithEmbeddingSerializer(serializers.ModelSerializer):
    class Meta:
        model = HadithEmbedding
        fields = ['id', 'hadith_number', 'embedding_turkish', 'embedding_arabic', 'created_at', 'updated_at']


class HadithChunkEmbeddingSerializer(serializers.ModelSerializer):
    class Meta:
        model = HadithChunkEmbedding
        fields = ['id', 'hadith_number', 'chunk_index', 'chunk_text', 'language', 'embedding', 'created_at']


class HadithSerializer(serializers.ModelSerializer):
    sources = HadithSourceSerializer(many=True, read_only=True)

    class Meta:
        model = Hadith
        fields = [
            'id',
            'hadith_number',
            'narrator',
            'narrator_arabic',
            'hadith_text_turkish',
            'hadith_text_arabic',
            'language',
            'sources',
            'created_at',
            'updated_at',
        ]
