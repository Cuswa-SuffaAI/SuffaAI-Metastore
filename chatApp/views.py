from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Hadith, HadithEmbedding, HadithSource, HadithChunkEmbedding
from .serializers import HadithSerializer, HadithEmbeddingSerializer, HadithSourceSerializer, HadithChunkEmbeddingSerializer
from .services import HadithService, EmbeddingService, ChunkEmbeddingService


class HadithViewSet(viewsets.ModelViewSet):
    """API endpoint for Hadith CRUD operations."""
    queryset = Hadith.objects.all()
    serializer_class = HadithSerializer

    @action(detail=False, methods=['get'])
    def search(self, request):
        """Search hadiths by narrator. Usage: /api/hadiths/search/?narrator=Enes"""
        narrator = request.query_params.get('narrator', '')
        if not narrator:
            return Response({'error': 'narrator parameter required'}, status=status.HTTP_400_BAD_REQUEST)

        hadiths = HadithService.search_by_narrator(narrator)
        serializer = self.get_serializer(hadiths, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def add_sources(self, request, pk=None):
        """Add sources to a hadith. POST /api/hadiths/{id}/add_sources/"""
        hadith = self.get_object()
        sources = request.data.get('sources', [])

        for source in sources:
            HadithSource.objects.create(
                hadith=hadith,
                name=source.get('name'),
                reference=source.get('reference')
            )

        serializer = self.get_serializer(hadith)
        return Response(serializer.data)

    @action(detail=False, methods=['post'])
    def bulk_create(self, request):
        """
        Bulk create hadiths with sources.
        POST /api/hadiths/bulk_create/

        Request body:
        [
            {
                "hadith_number": 15,
                "narrator": "Enes ibn Mâlik el Ensârî",
                "narrator_arabic": "عَنْ أَبِي حَمْزَةَ...",
                "hadith_text_turkish": "...",
                "hadith_text_arabic": "...",
                "language": "tr-ar",
                "sources": [
                    {"name": "Buhârî", "reference": "Deavât 4"},
                    {"name": "Müslim", "reference": "Tevbe 1, 7"}
                ]
            },
            ...
        ]
        """
        hadiths_data = request.data

        if not isinstance(hadiths_data, list):
            return Response(
                {'error': 'Request body must be a list of hadiths'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not hadiths_data:
            return Response(
                {'error': 'Empty list provided'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            result = HadithService.bulk_create_hadiths(hadiths_data)
            return Response(result, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HadithEmbeddingViewSet(viewsets.ModelViewSet):
    """API endpoint for HadithEmbedding CRUD operations."""
    queryset = HadithEmbedding.objects.all()
    serializer_class = HadithEmbeddingSerializer

    @action(detail=False, methods=['post'])
    def save_for_hadith(self, request):
        """Save embeddings for a hadith. POST /api/hadith-embeddings/save_for_hadith/"""
        hadith_id = request.data.get('hadith_id')
        turkish_embedding = request.data.get('embedding_turkish')
        arabic_embedding = request.data.get('embedding_arabic')

        try:
            hadith = Hadith.objects.get(id=hadith_id)
        except Hadith.DoesNotExist:
            return Response({'error': 'Hadith not found'}, status=status.HTTP_404_NOT_FOUND)

        embedding = EmbeddingService.save_embeddings(hadith, turkish_embedding, arabic_embedding)
        serializer = self.get_serializer(embedding)
        return Response(serializer.data)

    @action(detail=False, methods=['post'])
    def bulk_create(self, request):
        """
        Bulk create embeddings for hadiths.
        POST /api/hadith-embeddings/bulk_create/

        Request body:
        [
            {
                "hadith_number": 15,
                "embedding_turkish": [0.1, 0.2, ...],
                "embedding_arabic": [0.3, 0.4, ...]
            },
            ...
        ]
        """
        embeddings_data = request.data

        if not isinstance(embeddings_data, list):
            return Response(
                {'error': 'Request body must be a list of embeddings'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not embeddings_data:
            return Response(
                {'error': 'Empty list provided'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            result = EmbeddingService.bulk_create_embeddings(embeddings_data)

            if 'error' in result:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)

            return Response(result, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['post'])
    def search_similar(self, request):
        """
        Search similar hadiths using vector similarity.
        POST /api/hadith-embeddings/search_similar/

        Request body:
        {
            "embedding": [0.1, 0.2, ...],  # 768 dimensions
            "language": "turkish",          # or "arabic"
            "limit": 10                     # optional, default 10
        }
        """
        query_embedding = request.data.get('embedding')
        language = request.data.get('language', 'turkish')
        limit = request.data.get('limit', 10)

        if not query_embedding:
            return Response(
                {'error': 'embedding is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not isinstance(query_embedding, list) or len(query_embedding) != 768:
            return Response(
                {'error': 'embedding must be a list of 768 dimensions'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if language not in ['turkish', 'arabic']:
            return Response(
                {'error': 'language must be "turkish" or "arabic"'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            results = EmbeddingService.search_similar_with_hadith_details(
                query_embedding=query_embedding,
                language=language,
                limit=limit
            )
            return Response({
                'count': len(results),
                'language': language,
                'results': results
            })
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HadithSourceViewSet(viewsets.ModelViewSet):
    """API endpoint for HadithSource CRUD operations."""
    queryset = HadithSource.objects.all()
    serializer_class = HadithSourceSerializer


class HadithChunkEmbeddingViewSet(viewsets.ModelViewSet):
    """API endpoint for HadithChunkEmbedding CRUD operations."""
    queryset = HadithChunkEmbedding.objects.all()
    serializer_class = HadithChunkEmbeddingSerializer

    @action(detail=False, methods=['post'])
    def bulk_create(self, request):
        """
        Bulk create chunk embeddings for hadiths.
        POST /api/hadith-chunk-embeddings/bulk_create/

        Request body:
        [
            {
                "hadith_number": 15,
                "chunk_index": 0,
                "chunk_text": "...",
                "language": "turkish",
                "embedding": [0.1, 0.2, ...]
            },
            ...
        ]
        """
        chunks_data = request.data

        if not isinstance(chunks_data, list):
            return Response(
                {'error': 'Request body must be a list of chunk embeddings'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not chunks_data:
            return Response(
                {'error': 'Empty list provided'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            result = ChunkEmbeddingService.bulk_create_chunk_embeddings(chunks_data)

            if 'error' in result:
                return Response(result, status=status.HTTP_400_BAD_REQUEST)

            return Response(result, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['post'])
    def search_similar(self, request):
        """
        Search similar chunks using vector similarity.
        POST /api/hadith-chunk-embeddings/search_similar/

        Request body:
        {
            "embedding": [0.1, 0.2, ...],  # 768 dimensions
            "language": "turkish",          # or "arabic"
            "limit": 10                     # optional, default 10
        }
        """
        query_embedding = request.data.get('embedding')
        language = request.data.get('language', 'turkish')
        limit = request.data.get('limit', 10)

        if not query_embedding:
            return Response(
                {'error': 'embedding is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not isinstance(query_embedding, list) or len(query_embedding) != 768:
            return Response(
                {'error': 'embedding must be a list of 768 dimensions'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if language not in ['turkish', 'arabic']:
            return Response(
                {'error': 'language must be "turkish" or "arabic"'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            results = ChunkEmbeddingService.search_similar_with_details(
                query_embedding=query_embedding,
                language=language,
                limit=limit
            )
            return Response({
                'count': len(results),
                'language': language,
                'results': results
            })
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['delete'])
    def delete_for_hadith(self, request):
        """
        Delete all chunk embeddings for a specific hadith.
        DELETE /api/hadith-chunk-embeddings/delete_for_hadith/

        Request body:
        {
            "hadith_number": 15,
            "language": "turkish"  # optional, if not provided deletes all languages
        }
        """
        hadith_number = request.data.get('hadith_number')
        language = request.data.get('language')

        if not hadith_number:
            return Response(
                {'error': 'hadith_number is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            deleted_count = ChunkEmbeddingService.delete_chunks_for_hadith(
                hadith_number=hadith_number,
                language=language
            )
            return Response({
                'deleted_count': deleted_count,
                'hadith_number': hadith_number,
                'language': language
            })
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
