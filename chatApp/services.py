from django.db import transaction
from pgvector.django import CosineDistance, L2Distance, MaxInnerProduct
from .models import Hadith, HadithEmbedding, HadithSource, HadithChunkEmbedding
from .models import SiyerSection, SiyerChunkEmbedding


class HadithService:
    """Business logic for Hadith operations."""

    @staticmethod
    def create_hadith_with_sources(data: dict, sources: list[dict]) -> Hadith:
        """Create a hadith with its sources."""
        hadith = Hadith.objects.create(**data)
        for source in sources:
            HadithSource.objects.create(hadith=hadith, **source)
        return hadith

    @staticmethod
    @transaction.atomic
    def bulk_create_hadiths(hadiths_data: list[dict]) -> dict:
        """
        Bulk create hadiths with their sources.

        Expected format:
        [
            {
                "hadith_number": 15,
                "narrator": "...",
                "narrator_arabic": "...",
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
        created_hadiths = []
        all_sources = []
        null_hadith_numbers = 0

        # First pass: create Hadith objects
        hadith_objects = []
        sources_mapping = []  # Store sources for each hadith index

        for item in hadiths_data:
            # Hadis numarası kontrolü - geçersiz ise null olarak kaydet
            hadith_number = item.get('hadith_number')
            if isinstance(hadith_number, str):
                try:
                    hadith_number = int(hadith_number)
                except ValueError:
                    hadith_number = None
            if hadith_number is not None and hadith_number <= 0:
                hadith_number = None
            if hadith_number is None:
                null_hadith_numbers += 1

            sources = item.pop('sources', [])
            sources_mapping.append(sources)
            hadith_objects.append(Hadith(
                hadith_number=hadith_number,
                narrator=item.get('narrator'),
                narrator_arabic=item.get('narrator_arabic'),
                hadith_text_turkish=item.get('hadith_text_turkish'),
                hadith_text_arabic=item.get('hadith_text_arabic'),
                language=item.get('language', 'tr-ar'),
                potential_questions=item.get('potential_questions'),
                embedding_text=item.get('embedding_text'),
                priority_keywords=item.get('priority_keywords'),
            ))

        # Bulk create hadiths
        created_hadiths = Hadith.objects.bulk_create(hadith_objects)

        # Second pass: create sources with hadith references
        for hadith, sources in zip(created_hadiths, sources_mapping):
            for source in sources:
                all_sources.append(HadithSource(
                    hadith=hadith,
                    name=source.get('name'),
                    reference=source.get('reference'),
                ))

        # Bulk create sources
        if all_sources:
            HadithSource.objects.bulk_create(all_sources)

        return {
            'hadiths_created': len(created_hadiths),
            'sources_created': len(all_sources),
            'hadith_ids': [h.id for h in created_hadiths],
            'null_hadith_numbers': null_hadith_numbers,
        }

    @staticmethod
    def search_by_narrator(narrator: str):
        """Search hadiths by narrator name."""
        return Hadith.objects.filter(narrator__icontains=narrator)

    @staticmethod
    def get_hadith_with_embeddings(hadith_id: int):
        """Get hadith with its embeddings."""
        return Hadith.objects.select_related('embeddings').get(id=hadith_id)


class EmbeddingService:
    """Business logic for embedding operations."""

    @staticmethod
    def save_embeddings(hadith: Hadith, turkish_embedding: list, arabic_embedding: list) -> HadithEmbedding:
        """Save embeddings for a hadith."""
        embedding, created = HadithEmbedding.objects.update_or_create(
            hadith=hadith,
            defaults={
                'hadith_number': hadith.hadith_number,
                'embedding_turkish': turkish_embedding,
                'embedding_arabic': arabic_embedding,
            }
        )
        return embedding

    @staticmethod
    @transaction.atomic
    def bulk_create_embeddings(embeddings_data: list[dict]) -> dict:
        """
        Bulk create embeddings for hadiths.

        Expected format:
        [
            {
                "hadith_number": 15,
                "embedding_turkish": [0.1, 0.2, ...],  # 768 dimensions
                "embedding_arabic": [0.3, 0.4, ...]    # 768 dimensions
            },
            ...
        ]
        """
        # Get all hadith_numbers from request
        hadith_numbers = [item.get('hadith_number') for item in embeddings_data]

        # Fetch all hadiths in one query
        hadiths = Hadith.objects.filter(hadith_number__in=hadith_numbers)
        hadith_map = {h.hadith_number: h for h in hadiths}

        # Check for missing hadiths
        missing = set(hadith_numbers) - set(hadith_map.keys())
        if missing:
            return {
                'error': f'Hadiths not found for hadith_numbers: {list(missing)}',
                'embeddings_created': 0,
            }

        # Check for existing embeddings
        existing_embeddings = HadithEmbedding.objects.filter(
            hadith_number__in=hadith_numbers
        ).values_list('hadith_number', flat=True)
        existing_set = set(existing_embeddings)

        # Prepare embedding objects
        embedding_objects = []
        skipped = []

        for item in embeddings_data:
            hadith_number = item.get('hadith_number')

            # Skip if embedding already exists
            if hadith_number in existing_set:
                skipped.append(hadith_number)
                continue

            hadith = hadith_map.get(hadith_number)
            if hadith:
                embedding_objects.append(HadithEmbedding(
                    hadith=hadith,
                    hadith_number=hadith_number,
                    embedding_turkish=item.get('embedding_turkish'),
                    embedding_arabic=item.get('embedding_arabic'),
                ))

        # Bulk create
        created_embeddings = []
        if embedding_objects:
            created_embeddings = HadithEmbedding.objects.bulk_create(embedding_objects)

        return {
            'embeddings_created': len(created_embeddings),
            'skipped_existing': skipped,
            'embedding_ids': [e.id for e in created_embeddings],
        }

    @staticmethod
    def search_similar(
        query_embedding: list,
        language: str = 'turkish',
        limit: int = 10,
        distance_type: str = 'cosine'
    ):
        """
        Search similar hadiths using pgvector similarity search.

        Args:
            query_embedding: 768-dimensional vector
            language: 'turkish' or 'arabic'
            limit: max results to return
            distance_type: 'cosine', 'l2', or 'inner_product'

        Returns:
            QuerySet of HadithEmbedding with distance annotation
        """
        # Select the appropriate embedding field
        if language == 'turkish':
            embedding_field = 'embedding_turkish'
        else:
            embedding_field = 'embedding_arabic'

        # Select distance function
        if distance_type == 'cosine':
            distance_func = CosineDistance(embedding_field, query_embedding)
        elif distance_type == 'l2':
            distance_func = L2Distance(embedding_field, query_embedding)
        elif distance_type == 'inner_product':
            distance_func = MaxInnerProduct(embedding_field, query_embedding)
        else:
            distance_func = CosineDistance(embedding_field, query_embedding)

        # Query with distance annotation
        results = (
            HadithEmbedding.objects
            .exclude(**{f'{embedding_field}__isnull': True})
            .annotate(distance=distance_func)
            .order_by('distance')
            .select_related('hadith')
            [:limit]
        )

        return results

    @staticmethod
    def search_similar_with_hadith_details(
        query_embedding: list,
        language: str = 'turkish',
        limit: int = 10
    ):
        """
        Search similar hadiths and return full hadith details.

        Returns list of dicts with hadith info and similarity score.
        """
        results = EmbeddingService.search_similar(
            query_embedding=query_embedding,
            language=language,
            limit=limit,
            distance_type='cosine'
        )

        # Prefetch sources for all hadiths
        hadith_ids = [r.hadith.id for r in results]
        hadiths_with_sources = Hadith.objects.filter(id__in=hadith_ids).prefetch_related('sources')
        hadith_map = {h.id: h for h in hadiths_with_sources}

        return [
            {
                'hadith_number': r.hadith_number,
                'hadith_id': r.hadith.id,
                'narrator': r.hadith.narrator,
                'hadith_text_turkish': r.hadith.hadith_text_turkish,
                'hadith_text_arabic': r.hadith.hadith_text_arabic,
                'embedding_text': r.hadith.embedding_text,
                'priority_keywords': r.hadith.priority_keywords,
                'sources': [
                    {'name': s.name, 'reference': s.reference}
                    for s in hadith_map.get(r.hadith.id, r.hadith).sources.all()
                ],
                'distance': float(r.distance),
                'similarity': 1 - float(r.distance),
            }
            for r in results
        ]


class ChunkEmbeddingService:
    """Business logic for chunk-based embedding operations."""

    @staticmethod
    @transaction.atomic
    def bulk_create_chunk_embeddings(chunks_data: list[dict]) -> dict:
        """
        Bulk create chunk embeddings for hadiths.

        Expected format:
        [
            {
                "hadith_number": 15,
                "chunk_type": "embedding_text",  # or "potential_question"
                "chunk_index": 0,
                "chunk_text": "...",
                "embedding": [0.1, 0.2, ...]  # 768 dimensions
            },
            ...
        ]
        """
        # Get unique hadith_numbers from request
        hadith_numbers = list(set(item.get('hadith_number') for item in chunks_data))

        # Fetch all hadiths in one query
        hadiths = Hadith.objects.filter(hadith_number__in=hadith_numbers)
        hadith_map = {h.hadith_number: h for h in hadiths}

        # Check for missing hadiths
        missing = set(hadith_numbers) - set(hadith_map.keys())
        if missing:
            return {
                'error': f'Hadiths not found for hadith_numbers: {list(missing)}',
                'chunks_created': 0,
            }

        # Prepare chunk embedding objects
        chunk_objects = []
        for item in chunks_data:
            hadith_number = item.get('hadith_number')
            hadith = hadith_map.get(hadith_number)
            if hadith:
                chunk_objects.append(HadithChunkEmbedding(
                    hadith=hadith,
                    hadith_number=hadith_number,
                    chunk_type=item.get('chunk_type', 'embedding_text'),
                    chunk_index=item.get('chunk_index', 0),
                    chunk_text=item.get('chunk_text'),
                    embedding=item.get('embedding'),
                ))

        # Bulk create
        created_chunks = []
        if chunk_objects:
            created_chunks = HadithChunkEmbedding.objects.bulk_create(chunk_objects)

        return {
            'chunks_created': len(created_chunks),
            'chunk_ids': [c.id for c in created_chunks],
        }

    @staticmethod
    def delete_chunks_for_hadith(hadith_number: int, chunk_type: str = None) -> int:
        """Delete existing chunks for a hadith before re-creating."""
        queryset = HadithChunkEmbedding.objects.filter(hadith_number=hadith_number)
        if chunk_type:
            queryset = queryset.filter(chunk_type=chunk_type)
        count, _ = queryset.delete()
        return count

    @staticmethod
    def search_similar_chunks(
        query_embedding: list,
        chunk_type: str = None,
        limit: int = 10
    ):
        """
        Search similar chunks using pgvector similarity search.

        Args:
            query_embedding: 768-dimensional vector
            chunk_type: 'embedding_text' or 'potential_question' (None for all)
            limit: max results to return

        Returns:
            QuerySet of HadithChunkEmbedding with distance annotation
        """
        distance_func = CosineDistance('embedding', query_embedding)

        queryset = HadithChunkEmbedding.objects.all()
        if chunk_type:
            queryset = queryset.filter(chunk_type=chunk_type)

        results = (
            queryset
            .annotate(distance=distance_func)
            .order_by('distance')
            .select_related('hadith')
            .prefetch_related('hadith__sources')
            [:limit]
        )

        return results

    @staticmethod
    def search_with_combined_score(
        query_embedding: list,
        limit: int = 10,
        embedding_text_weight: float = 0.4,
        question_weight: float = 0.6
    ):
        """
        Search hadiths using combined score from embedding_text and potential_questions.

        The combined score is calculated as:
        - Best embedding_text similarity * embedding_text_weight
        - Best potential_question similarity * question_weight

        Args:
            query_embedding: 768-dimensional vector
            limit: max results to return
            embedding_text_weight: weight for embedding_text similarity (default 0.4)
            question_weight: weight for potential_question similarity (default 0.6)

        Returns:
            List of dicts with hadith info and combined score
        """
        distance_func = CosineDistance('embedding', query_embedding)

        # Get all chunks with distances
        all_chunks = (
            HadithChunkEmbedding.objects
            .annotate(distance=distance_func)
            .select_related('hadith')
            .prefetch_related('hadith__sources')
            .order_by('distance')
        )

        # Group by hadith_number and calculate combined scores
        hadith_scores = {}

        for chunk in all_chunks:
            hadith_num = chunk.hadith_number
            similarity = 1 - float(chunk.distance)  # Convert distance to similarity

            if hadith_num not in hadith_scores:
                hadith_scores[hadith_num] = {
                    'hadith': chunk.hadith,
                    'embedding_text_similarity': 0.0,
                    'best_question_similarity': 0.0,
                    'matching_questions': [],
                }

            if chunk.chunk_type == 'embedding_text':
                hadith_scores[hadith_num]['embedding_text_similarity'] = similarity
            elif chunk.chunk_type == 'potential_question':
                # Keep track of best question match
                if similarity > hadith_scores[hadith_num]['best_question_similarity']:
                    hadith_scores[hadith_num]['best_question_similarity'] = similarity
                # Also store matching questions with high similarity
                if similarity > 0.5:
                    hadith_scores[hadith_num]['matching_questions'].append({
                        'question': chunk.chunk_text,
                        'similarity': similarity
                    })

        # Calculate combined scores
        results = []
        for hadith_num, data in hadith_scores.items():
            combined_score = (
                data['embedding_text_similarity'] * embedding_text_weight +
                data['best_question_similarity'] * question_weight
            )

            hadith = data['hadith']
            results.append({
                'hadith_number': hadith_num,
                'hadith_id': hadith.id,
                'narrator': hadith.narrator,
                'hadith_text_turkish': hadith.hadith_text_turkish,
                'hadith_text_arabic': hadith.hadith_text_arabic,
                'embedding_text': hadith.embedding_text,
                'priority_keywords': hadith.priority_keywords,
                'sources': [
                    {'source_name': s.name, 'reference': s.reference}
                    for s in hadith.sources.all()
                ],
                'scores': {
                    'combined': combined_score,
                    'embedding_text_similarity': data['embedding_text_similarity'],
                    'best_question_similarity': data['best_question_similarity'],
                },
                'matching_questions': sorted(
                    data['matching_questions'],
                    key=lambda x: x['similarity'],
                    reverse=True
                )[:3]  # Top 3 matching questions
            })

        # Sort by combined score and return top results
        results.sort(key=lambda x: x['scores']['combined'], reverse=True)
        return results[:limit]

    @staticmethod
    def search_similar_with_details(
        query_embedding: list,
        chunk_type: str = None,
        limit: int = 10
    ):
        """
        Search similar chunks and return full details including hadith info.

        Returns list of dicts with chunk and hadith info.
        """
        results = ChunkEmbeddingService.search_similar_chunks(
            query_embedding=query_embedding,
            chunk_type=chunk_type,
            limit=limit
        )

        # Prefetch sources to avoid N+1 queries
        hadith_ids = [r.hadith_id for r in results]
        hadiths_with_sources = Hadith.objects.filter(id__in=hadith_ids).prefetch_related('sources')
        hadith_map = {h.id: h for h in hadiths_with_sources}

        return [
            {
                'chunk_id': r.id,
                'hadith_number': r.hadith_number,
                'chunk_type': r.chunk_type,
                'chunk_index': r.chunk_index,
                'chunk_text': r.chunk_text,
                'distance': float(r.distance),
                'similarity': 1 - float(r.distance),
                'hadith': {
                    'id': r.hadith.id,
                    'narrator': r.hadith.narrator,
                    'hadith_text_turkish': r.hadith.hadith_text_turkish,
                    'hadith_text_arabic': r.hadith.hadith_text_arabic,
                    'embedding_text': r.hadith.embedding_text,
                    'priority_keywords': r.hadith.priority_keywords,
                    'sources': [
                        {'source_name': s.name, 'reference': s.reference}
                        for s in hadith_map.get(r.hadith_id, r.hadith).sources.all()
                    ]
                }
            }
            for r in results
        ]


# ============================================================
# SIYER SERVİSLERİ
# ============================================================

class SiyerService:
    """Business logic for Siyer operations."""

    @staticmethod
    def create_section(data: dict) -> SiyerSection:
        """Create a single siyer section from gpt_metadata.json format."""
        return SiyerSection.objects.create(
            section_id=data.get('id'),
            text=data.get('text'),
            pages=data.get('pages'),
            summary_short=data.get('content_information', {}).get('summary_short'),
            main_theme=data.get('content_information', {}).get('main_theme'),
            potential_questions=data.get('potential_questions'),
            query_intents=data.get('retrieval_hints', {}).get('query_intents'),
        )

    @staticmethod
    @transaction.atomic
    def bulk_create_sections(sections_data: list[dict]) -> dict:
        """
        Bulk create siyer sections.

        Expected format (gpt_metadata.json):
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
        section_objects = []
        skipped = []

        # Check for existing section_ids
        incoming_ids = [item.get('id') for item in sections_data]
        existing_ids = set(
            SiyerSection.objects.filter(section_id__in=incoming_ids)
            .values_list('section_id', flat=True)
        )

        for item in sections_data:
            section_id = item.get('id')
            if section_id in existing_ids:
                skipped.append(section_id)
                continue

            content_info = item.get('content_information', {})
            retrieval_hints = item.get('retrieval_hints', {})

            section_objects.append(SiyerSection(
                section_id=section_id,
                text=item.get('text'),
                pages=item.get('pages'),
                summary_short=content_info.get('summary_short'),
                main_theme=content_info.get('main_theme'),
                potential_questions=item.get('potential_questions'),
                query_intents=retrieval_hints.get('query_intents'),
            ))

        created_sections = []
        if section_objects:
            created_sections = SiyerSection.objects.bulk_create(section_objects)

        return {
            'sections_created': len(created_sections),
            'skipped_existing': skipped,
            'section_ids': [s.section_id for s in created_sections],
        }

    @staticmethod
    def get_section_by_id(section_id: str) -> SiyerSection:
        """Get a siyer section by its section_id."""
        return SiyerSection.objects.get(section_id=section_id)

    @staticmethod
    def search_by_theme(theme: str):
        """Search siyer sections by main theme."""
        return SiyerSection.objects.filter(main_theme__icontains=theme)

    @staticmethod
    def search_by_page(page_number: int):
        """Search siyer sections that contain a specific page."""
        return SiyerSection.objects.filter(pages__contains=[page_number])


class SiyerChunkEmbeddingService:
    """Business logic for siyer chunk-based embedding operations."""

    @staticmethod
    @transaction.atomic
    def bulk_create_chunk_embeddings(chunks_data: list[dict]) -> dict:
        """
        Bulk create chunk embeddings for siyer sections.

        Expected format:
        [
            {
                "section_id": "siyer_1b257eea63da",
                "chunk_type": "summary",  # or "potential_question", "query_intent"
                "chunk_index": 0,
                "chunk_text": "...",
                "embedding": [0.1, 0.2, ...]  # 768 dimensions
            },
            ...
        ]
        """
        # Get unique section_ids
        section_ids = list(set(item.get('section_id') for item in chunks_data))

        # Fetch all sections in one query
        sections = SiyerSection.objects.filter(section_id__in=section_ids)
        section_map = {s.section_id: s for s in sections}

        # Check for missing sections
        missing = set(section_ids) - set(section_map.keys())
        if missing:
            return {
                'error': f'Sections not found for section_ids: {list(missing)}',
                'chunks_created': 0,
            }

        # Prepare chunk objects
        chunk_objects = []
        for item in chunks_data:
            section_id = item.get('section_id')
            section = section_map.get(section_id)
            if section:
                chunk_objects.append(SiyerChunkEmbedding(
                    section=section,
                    section_code=section_id,
                    chunk_type=item.get('chunk_type'),
                    chunk_index=item.get('chunk_index', 0),
                    chunk_text=item.get('chunk_text'),
                    embedding=item.get('embedding'),
                ))

        created_chunks = []
        if chunk_objects:
            created_chunks = SiyerChunkEmbedding.objects.bulk_create(chunk_objects)

        return {
            'chunks_created': len(created_chunks),
            'chunk_ids': [c.id for c in created_chunks],
        }

    @staticmethod
    def delete_chunks_for_section(section_id: str, chunk_type: str = None) -> int:
        """Delete existing chunks for a section before re-creating."""
        queryset = SiyerChunkEmbedding.objects.filter(section_code=section_id)
        if chunk_type:
            queryset = queryset.filter(chunk_type=chunk_type)
        count, _ = queryset.delete()
        return count

    @staticmethod
    def search_similar_chunks(
        query_embedding: list,
        chunk_type: str = None,
        limit: int = 10
    ):
        """
        Search similar siyer chunks using pgvector similarity search.

        Args:
            query_embedding: 768-dimensional vector
            chunk_type: 'summary', 'potential_question', 'query_intent' (None for all)
            limit: max results to return
        """
        distance_func = CosineDistance('embedding', query_embedding)

        queryset = SiyerChunkEmbedding.objects.all()
        if chunk_type:
            queryset = queryset.filter(chunk_type=chunk_type)

        results = (
            queryset
            .annotate(distance=distance_func)
            .order_by('distance')
            .select_related('section')
            [:limit]
        )

        return results

    @staticmethod
    def search_with_combined_score(
        query_embedding: list,
        limit: int = 10,
        summary_weight: float = 0.3,
        question_weight: float = 0.5,
        intent_weight: float = 0.2
    ):
        """
        Search siyer sections using combined score from all chunk types.

        Args:
            query_embedding: 768-dimensional vector
            limit: max results to return
            summary_weight: weight for summary similarity
            question_weight: weight for potential_question similarity
            intent_weight: weight for query_intent similarity
        """
        distance_func = CosineDistance('embedding', query_embedding)

        all_chunks = (
            SiyerChunkEmbedding.objects
            .annotate(distance=distance_func)
            .select_related('section')
            .order_by('distance')
        )

        # Group by section_code and calculate combined scores
        section_scores = {}

        for chunk in all_chunks:
            sid = chunk.section_code
            similarity = 1 - float(chunk.distance)

            if sid not in section_scores:
                section_scores[sid] = {
                    'section': chunk.section,
                    'summary_similarity': 0.0,
                    'best_question_similarity': 0.0,
                    'best_intent_similarity': 0.0,
                    'matching_questions': [],
                }

            if chunk.chunk_type == 'summary':
                section_scores[sid]['summary_similarity'] = similarity
            elif chunk.chunk_type == 'potential_question':
                if similarity > section_scores[sid]['best_question_similarity']:
                    section_scores[sid]['best_question_similarity'] = similarity
                if similarity > 0.5:
                    section_scores[sid]['matching_questions'].append({
                        'question': chunk.chunk_text,
                        'similarity': similarity
                    })
            elif chunk.chunk_type == 'query_intent':
                if similarity > section_scores[sid]['best_intent_similarity']:
                    section_scores[sid]['best_intent_similarity'] = similarity

        # Calculate combined scores
        results = []
        for sid, data in section_scores.items():
            combined_score = (
                data['summary_similarity'] * summary_weight +
                data['best_question_similarity'] * question_weight +
                data['best_intent_similarity'] * intent_weight
            )

            section = data['section']
            results.append({
                'section_id': sid,
                'text': section.text,
                'pages': section.pages,
                'summary_short': section.summary_short,
                'main_theme': section.main_theme,
                'potential_questions': section.potential_questions,
                'query_intents': section.query_intents,
                'scores': {
                    'combined': combined_score,
                    'summary_similarity': data['summary_similarity'],
                    'best_question_similarity': data['best_question_similarity'],
                    'best_intent_similarity': data['best_intent_similarity'],
                },
                'matching_questions': sorted(
                    data['matching_questions'],
                    key=lambda x: x['similarity'],
                    reverse=True
                )[:3]
            })

        results.sort(key=lambda x: x['scores']['combined'], reverse=True)
        return results[:limit]

    @staticmethod
    def search_similar_with_details(
        query_embedding: list,
        chunk_type: str = None,
        limit: int = 10
    ):
        """
        Search similar siyer chunks and return full section details.
        """
        results = SiyerChunkEmbeddingService.search_similar_chunks(
            query_embedding=query_embedding,
            chunk_type=chunk_type,
            limit=limit
        )

        return [
            {
                'chunk_id': r.id,
                'section_id': r.section_code,
                'chunk_type': r.chunk_type,
                'chunk_index': r.chunk_index,
                'chunk_text': r.chunk_text,
                'distance': float(r.distance),
                'similarity': 1 - float(r.distance),
                'section': {
                    'text': r.section.text,
                    'pages': r.section.pages,
                    'summary_short': r.section.summary_short,
                    'main_theme': r.section.main_theme,
                    'potential_questions': r.section.potential_questions,
                    'query_intents': r.section.query_intents,
                }
            }
            for r in results
        ]
