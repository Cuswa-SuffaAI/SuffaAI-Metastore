from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Hadith, HadithEmbedding, HadithSource, HadithChunkEmbedding
from .models import SiyerSection, SiyerChunkEmbedding
from .models import FetvaQuestion, FetvaChunkEmbedding
from .serializers import HadithSerializer, HadithEmbeddingSerializer, HadithSourceSerializer, HadithChunkEmbeddingSerializer
from .serializers import SiyerSectionSerializer, SiyerChunkEmbeddingSerializer
from .serializers import FetvaQuestionSerializer, FetvaChunkEmbeddingSerializer
from .services import HadithService, EmbeddingService, ChunkEmbeddingService
from .services import SiyerService, SiyerChunkEmbeddingService
from .services import FetvaService, FetvaChunkEmbeddingService


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
            import traceback
            traceback.print_exc()
            return Response(
                {'error': str(e), 'traceback': traceback.format_exc()},
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
                "chunk_type": "embedding_text",  # or "potential_question"
                "chunk_index": 0,
                "chunk_text": "...",
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
            "chunk_type": "embedding_text", # optional: "embedding_text", "potential_question", or null for all
            "limit": 10                     # optional, default 10
        }
        """
        query_embedding = request.data.get('embedding')
        chunk_type = request.data.get('chunk_type')
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

        if chunk_type and chunk_type not in ['embedding_text', 'potential_question']:
            return Response(
                {'error': 'chunk_type must be "embedding_text" or "potential_question"'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            results = ChunkEmbeddingService.search_similar_with_details(
                query_embedding=query_embedding,
                chunk_type=chunk_type,
                limit=limit
            )
            return Response({
                'count': len(results),
                'chunk_type': chunk_type,
                'results': results
            })
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['post'])
    def search_combined(self, request):
        """
        Search hadiths using combined score from embedding_text and potential_questions.
        POST /api/hadith-chunk-embeddings/search_combined/

        The combined score is calculated as:
        - Best embedding_text similarity * embedding_text_weight
        - Best potential_question similarity * question_weight

        Request body:
        {
            "embedding": [0.1, 0.2, ...],    # 768 dimensions
            "limit": 10,                      # optional, default 10
            "embedding_text_weight": 0.4,     # optional, default 0.4
            "question_weight": 0.6            # optional, default 0.6
        }

        Response includes:
        - Combined score from both embedding_text and potential_questions
        - Individual similarity scores for each type
        - Top 3 matching questions per hadith
        """
        query_embedding = request.data.get('embedding')
        limit = request.data.get('limit', 10)
        embedding_text_weight = request.data.get('embedding_text_weight', 0.4)
        question_weight = request.data.get('question_weight', 0.6)

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

        try:
            results = ChunkEmbeddingService.search_with_combined_score(
                query_embedding=query_embedding,
                limit=limit,
                embedding_text_weight=embedding_text_weight,
                question_weight=question_weight
            )
            return Response({
                'count': len(results),
                'weights': {
                    'embedding_text': embedding_text_weight,
                    'question': question_weight
                },
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
            "chunk_type": "embedding_text"  # optional, if not provided deletes all types
        }
        """
        hadith_number = request.data.get('hadith_number')
        chunk_type = request.data.get('chunk_type')

        if not hadith_number:
            return Response(
                {'error': 'hadith_number is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            deleted_count = ChunkEmbeddingService.delete_chunks_for_hadith(
                hadith_number=hadith_number,
                chunk_type=chunk_type
            )
            return Response({
                'deleted_count': deleted_count,
                'hadith_number': hadith_number,
                'chunk_type': chunk_type
            })
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ============================================================
# SIYER VIEWS
# ============================================================

class SiyerSectionViewSet(viewsets.ModelViewSet):
    """API endpoint for SiyerSection CRUD operations."""
    queryset = SiyerSection.objects.all()
    serializer_class = SiyerSectionSerializer

    @action(detail=False, methods=['get'])
    def search(self, request):
        """
        Search siyer sections by theme or page.
        GET /api/siyer-sections/search/?theme=dogum
        GET /api/siyer-sections/search/?page=15
        """
        theme = request.query_params.get('theme')
        page = request.query_params.get('page')

        if theme:
            sections = SiyerService.search_by_theme(theme)
        elif page:
            try:
                sections = SiyerService.search_by_page(int(page))
            except ValueError:
                return Response({'error': 'page must be an integer'}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(
                {'error': 'theme or page parameter required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        serializer = self.get_serializer(sections, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['post'])
    def bulk_create(self, request):
        """
        Bulk create siyer sections from gpt_metadata.json format.
        POST /api/siyer-sections/bulk_create/

        Request body (all_metadata.json format):
        [
            {
                "id": "siyer_1b257eea63da",
                "text": "...",
                "pages": [12, 13, 14],
                "content_information": {
                    "summary_short": "...",
                    "main_theme": "..."
                },
                "potential_questions": ["Soru 1?", ...],
                "retrieval_hints": {
                    "query_intents": ["anahtar1", ...]
                }
            },
            ...
        ]
        """
        sections_data = request.data

        if not isinstance(sections_data, list):
            return Response(
                {'error': 'Request body must be a list of sections'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not sections_data:
            return Response(
                {'error': 'Empty list provided'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            result = SiyerService.bulk_create_sections(sections_data)
            return Response(result, status=status.HTTP_201_CREATED)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response(
                {'error': str(e), 'traceback': traceback.format_exc()},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SiyerChunkEmbeddingViewSet(viewsets.ModelViewSet):
    """API endpoint for SiyerChunkEmbedding CRUD operations."""
    queryset = SiyerChunkEmbedding.objects.all()
    serializer_class = SiyerChunkEmbeddingSerializer

    @action(detail=False, methods=['post'])
    def bulk_create(self, request):
        """
        Bulk create chunk embeddings for siyer sections.
        POST /api/siyer-chunk-embeddings/bulk_create/

        Request body:
        [
            {
                "section_id": "siyer_1b257eea63da",
                "chunk_type": "summary",
                "chunk_index": 0,
                "chunk_text": "...",
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
            result = SiyerChunkEmbeddingService.bulk_create_chunk_embeddings(chunks_data)

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
        Search similar siyer chunks using vector similarity.
        POST /api/siyer-chunk-embeddings/search_similar/

        Request body:
        {
            "embedding": [0.1, 0.2, ...],
            "chunk_type": "summary",        # optional
            "limit": 10                      # optional
        }
        """
        query_embedding = request.data.get('embedding')
        chunk_type = request.data.get('chunk_type')
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

        if chunk_type and chunk_type not in ['summary', 'potential_question', 'query_intent']:
            return Response(
                {'error': 'chunk_type must be "summary", "potential_question", or "query_intent"'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            results = SiyerChunkEmbeddingService.search_similar_with_details(
                query_embedding=query_embedding,
                chunk_type=chunk_type,
                limit=limit
            )
            return Response({
                'count': len(results),
                'chunk_type': chunk_type,
                'results': results
            })
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['post'])
    def search_combined(self, request):
        """
        Search siyer sections using combined score from all chunk types.
        POST /api/siyer-chunk-embeddings/search_combined/

        Request body:
        {
            "embedding": [0.1, 0.2, ...],
            "limit": 10,
            "summary_weight": 0.3,
            "question_weight": 0.5,
            "intent_weight": 0.2
        }
        """
        query_embedding = request.data.get('embedding')
        limit = request.data.get('limit', 10)
        summary_weight = request.data.get('summary_weight', 0.3)
        question_weight = request.data.get('question_weight', 0.5)
        intent_weight = request.data.get('intent_weight', 0.2)

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

        try:
            results = SiyerChunkEmbeddingService.search_with_combined_score(
                query_embedding=query_embedding,
                limit=limit,
                summary_weight=summary_weight,
                question_weight=question_weight,
                intent_weight=intent_weight
            )
            return Response({
                'count': len(results),
                'weights': {
                    'summary': summary_weight,
                    'question': question_weight,
                    'intent': intent_weight
                },
                'results': results
            })
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['delete'])
    def delete_for_section(self, request):
        """
        Delete all chunk embeddings for a specific section.
        DELETE /api/siyer-chunk-embeddings/delete_for_section/

        Request body:
        {
            "section_id": "siyer_1b257eea63da",
            "chunk_type": "summary"  # optional
        }
        """
        section_id = request.data.get('section_id')
        chunk_type = request.data.get('chunk_type')

        if not section_id:
            return Response(
                {'error': 'section_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            deleted_count = SiyerChunkEmbeddingService.delete_chunks_for_section(
                section_id=section_id,
                chunk_type=chunk_type
            )
            return Response({
                'deleted_count': deleted_count,
                'section_id': section_id,
                'chunk_type': chunk_type
            })
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ============================================================
# FETVA VIEWS
# ============================================================

class FetvaQuestionViewSet(viewsets.ModelViewSet):
    """API endpoint for FetvaQuestion CRUD operations."""
    queryset = FetvaQuestion.objects.all()
    serializer_class = FetvaQuestionSerializer

    @action(detail=False, methods=['get'])
    def search(self, request):
        """
        Search fetva questions by subject or page.
        GET /api/fetva-questions/search/?subject=itikat
        GET /api/fetva-questions/search/?page=56
        """
        subject = request.query_params.get('subject')
        page = request.query_params.get('page')

        if subject:
            questions = FetvaService.search_by_subject(subject)
        elif page:
            try:
                questions = FetvaService.search_by_page(int(page))
            except ValueError:
                return Response({'error': 'page must be an integer'}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(
                {'error': 'subject or page parameter required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        serializer = self.get_serializer(questions, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['post'])
    def bulk_create(self, request):
        """
        Bulk create fetva questions.
        POST /api/fetva-questions/bulk_create/

        Request body:
        [
            {
                "id": "e0ca3e868553",
                "question": "Allah nerededir?",
                "answer": "...",
                "page": 56,
                "subject": "itikat",
                "paraphrase_questions": ["Soru 1?", ...]
            },
            ...
        ]
        """
        questions_data = request.data

        if not isinstance(questions_data, list):
            return Response(
                {'error': 'Request body must be a list of questions'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not questions_data:
            return Response(
                {'error': 'Empty list provided'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            result = FetvaService.bulk_create_questions(questions_data)
            return Response(result, status=status.HTTP_201_CREATED)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response(
                {'error': str(e), 'traceback': traceback.format_exc()},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class FetvaChunkEmbeddingViewSet(viewsets.ModelViewSet):
    """API endpoint for FetvaChunkEmbedding CRUD operations."""
    queryset = FetvaChunkEmbedding.objects.all()
    serializer_class = FetvaChunkEmbeddingSerializer

    @action(detail=False, methods=['post'])
    def bulk_create(self, request):
        """
        Bulk create chunk embeddings for fetva questions.
        POST /api/fetva-chunk-embeddings/bulk_create/

        Request body:
        [
            {
                "fetva_id": "e0ca3e868553",
                "chunk_type": "question",  # or "paraphrase_question"
                "chunk_index": 0,
                "chunk_text": "...",
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
            result = FetvaChunkEmbeddingService.bulk_create_chunk_embeddings(chunks_data)

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
        Search similar fetva chunks using vector similarity.
        POST /api/fetva-chunk-embeddings/search_similar/

        Request body:
        {
            "embedding": [0.1, 0.2, ...],
            "chunk_type": "question",  # optional: "question", "paraphrase_question", or null for all
            "limit": 10                # optional
        }
        """
        query_embedding = request.data.get('embedding')
        chunk_type = request.data.get('chunk_type')
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

        if chunk_type and chunk_type not in ['question', 'paraphrase_question']:
            return Response(
                {'error': 'chunk_type must be "question" or "paraphrase_question"'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            results = FetvaChunkEmbeddingService.search_similar_with_details(
                query_embedding=query_embedding,
                chunk_type=chunk_type,
                limit=limit
            )
            return Response({
                'count': len(results),
                'chunk_type': chunk_type,
                'results': results
            })
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['post'])
    def search_combined(self, request):
        """
        Search fetva questions using combined score from question and paraphrase chunks.
        POST /api/fetva-chunk-embeddings/search_combined/

        The combined score is:
        - best_question_similarity * question_weight      (higher impact, default 0.7)
        - best_paraphrase_similarity * paraphrase_weight  (lower impact, default 0.3)

        Request body:
        {
            "embedding": [0.1, 0.2, ...],
            "limit": 10,
            "question_weight": 0.7,    # optional
            "paraphrase_weight": 0.3   # optional
        }
        """
        query_embedding = request.data.get('embedding')
        limit = request.data.get('limit', 10)
        question_weight = request.data.get('question_weight', 0.7)
        paraphrase_weight = request.data.get('paraphrase_weight', 0.3)

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

        try:
            results = FetvaChunkEmbeddingService.search_with_combined_score(
                query_embedding=query_embedding,
                limit=limit,
                question_weight=question_weight,
                paraphrase_weight=paraphrase_weight,
            )
            return Response({
                'count': len(results),
                'weights': {
                    'question': question_weight,
                    'paraphrase': paraphrase_weight,
                },
                'results': results
            })
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['delete'])
    def delete_for_fetva(self, request):
        """
        Delete all chunk embeddings for a specific fetva question.
        DELETE /api/fetva-chunk-embeddings/delete_for_fetva/

        Request body:
        {
            "fetva_id": "e0ca3e868553",
            "chunk_type": "question"  # optional
        }
        """
        fetva_id = request.data.get('fetva_id')
        chunk_type = request.data.get('chunk_type')

        if not fetva_id:
            return Response(
                {'error': 'fetva_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            deleted_count = FetvaChunkEmbeddingService.delete_chunks_for_fetva(
                fetva_id=fetva_id,
                chunk_type=chunk_type
            )
            return Response({
                'deleted_count': deleted_count,
                'fetva_id': fetva_id,
                'chunk_type': chunk_type
            })
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
