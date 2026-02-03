from rest_framework import serializers
from .models import Hadith, HadithEmbedding, HadithSource, HadithChunkEmbedding
from .models import SiyerSection, SiyerChunkEmbedding


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
        fields = ['id', 'hadith_number', 'chunk_type', 'chunk_index', 'chunk_text', 'embedding', 'created_at']


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
            'potential_questions',
            'embedding_text',
            'priority_keywords',
            'sources',
            'created_at',
            'updated_at',
        ]


# ============================================================
# SIYER SERIALIZERS
# ============================================================

class SiyerChunkEmbeddingSerializer(serializers.ModelSerializer):
    class Meta:
        model = SiyerChunkEmbedding
        fields = ['id', 'section_code', 'chunk_type', 'chunk_index', 'chunk_text', 'embedding', 'created_at']


class SiyerSectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = SiyerSection
        fields = [
            'id',
            'section_id',
            'text',
            'pages',
            'summary_short',
            'main_theme',
            'potential_questions',
            'query_intents',
            'created_at',
            'updated_at',
        ]
