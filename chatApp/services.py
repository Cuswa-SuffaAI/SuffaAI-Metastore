from django.db import transaction
from pgvector.django import CosineDistance, L2Distance, MaxInnerProduct
from .models import Hadith, HadithEmbedding, HadithSource, HadithChunkEmbedding


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

        # First pass: create Hadith objects
        hadith_objects = []
        sources_mapping = []  # Store sources for each hadith index

        for item in hadiths_data:
            sources = item.pop('sources', [])
            sources_mapping.append(sources)
            hadith_objects.append(Hadith(
                hadith_number=item.get('hadith_number'),
                narrator=item.get('narrator'),
                narrator_arabic=item.get('narrator_arabic'),
                hadith_text_turkish=item.get('hadith_text_turkish'),
                hadith_text_arabic=item.get('hadith_text_arabic'),
                language=item.get('language', 'tr-ar'),
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

        return [
            {
                'hadith_number': r.hadith_number,
                'hadith_id': r.hadith.id,
                'narrator': r.hadith.narrator,
                'hadith_text_turkish': r.hadith.hadith_text_turkish,
                'hadith_text_arabic': r.hadith.hadith_text_arabic,
                'distance': float(r.distance),
                'similarity': 1 - float(r.distance),  # Cosine similarity = 1 - cosine distance
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
                "chunk_index": 0,
                "chunk_text": "...",
                "language": "turkish",
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
                    chunk_index=item.get('chunk_index'),
                    chunk_text=item.get('chunk_text'),
                    language=item.get('language'),
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
    def delete_chunks_for_hadith(hadith_number: int, language: str = None) -> int:
        """Delete existing chunks for a hadith before re-creating."""
        queryset = HadithChunkEmbedding.objects.filter(hadith_number=hadith_number)
        if language:
            queryset = queryset.filter(language=language)
        count, _ = queryset.delete()
        return count

    @staticmethod
    def search_similar_chunks(
        query_embedding: list,
        language: str = 'turkish',
        limit: int = 10
    ):
        """
        Search similar chunks using pgvector similarity search.

        Args:
            query_embedding: 768-dimensional vector
            language: 'turkish' or 'arabic'
            limit: max results to return

        Returns:
            QuerySet of HadithChunkEmbedding with distance annotation
        """
        distance_func = CosineDistance('embedding', query_embedding)

        results = (
            HadithChunkEmbedding.objects
            .filter(language=language)
            .annotate(distance=distance_func)
            .order_by('distance')
            .select_related('hadith')
            .prefetch_related('hadith__sources')
            [:limit]
        )

        return results

    @staticmethod
    def search_similar_with_details(
        query_embedding: list,
        language: str = 'turkish',
        limit: int = 10
    ):
        """
        Search similar chunks and return full details including hadith info.

        Returns list of dicts with chunk and hadith info.
        """
        results = ChunkEmbeddingService.search_similar_chunks(
            query_embedding=query_embedding,
            language=language,
            limit=limit
        )

        return [
            {
                'chunk_id': r.id,
                'hadith_number': r.hadith_number,
                'chunk_index': r.chunk_index,
                'chunk_text': r.chunk_text,
                'language': r.language,
                'distance': float(r.distance),
                'hadith': {
                    'id': r.hadith.id,
                    'narrator': r.hadith.narrator,
                    'hadith_text_turkish': r.hadith.hadith_text_turkish,
                    'hadith_text_arabic': r.hadith.hadith_text_arabic,
                    'sources': [
                        {'source_name': s.name}
                        for s in r.hadith.sources.all()
                    ]
                }
            }
            for r in results
        ]
