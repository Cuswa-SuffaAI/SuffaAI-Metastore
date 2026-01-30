from django.db import models
from pgvector.django import VectorField, HnswIndex

class ApiEndpoint(models.Model):
    """API endpoints table - stores API endpoint information with embeddings."""
    path = models.TextField(null=True, blank=True)
    method = models.TextField(null=True, blank=True)
    summary = models.TextField(null=True, blank=True)
    parameters = models.JSONField(null=True, blank=True)
    responses = models.JSONField(null=True, blank=True)
    embedding = VectorField(dimensions=768, null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'api_endpoints'

    def __str__(self):
        return f"{self.method} {self.path}"


class DocsChunk(models.Model):
    """Documentation chunks table - stores document chunks with embeddings."""
    text = models.TextField(null=True, blank=True)
    metadata = models.JSONField(null=True, blank=True)
    embedding = VectorField(dimensions=768, null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'docs_chunks'
        indexes = [
            models.Index(fields=['last_updated'], name='idx_docs_chunks_last_updated'),
        ]

    def __str__(self):
        return self.text[:50] if self.text else f"Chunk {self.id}"


class SystemEntity(models.Model):
    """System entities table - stores system entity definitions with embeddings."""
    name = models.TextField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    metadata = models.JSONField(null=True, blank=True)
    embedding = VectorField(dimensions=768, null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'system_entities'
        verbose_name_plural = 'System entities'
        indexes = [
            models.Index(fields=['last_updated'], name='idx_system_entities_last_upd'),
        ]

    def __str__(self):
        return self.name or f"Entity {self.id}"


class SystemField(models.Model):
    """System fields table - stores field definitions for system entities."""
    entity = models.ForeignKey(
        SystemEntity,
        on_delete=models.CASCADE,
        related_name='fields',
        null=True,
        blank=True
    )
    name = models.TextField(null=True, blank=True)
    type = models.TextField(null=True, blank=True)
    options = models.JSONField(null=True, blank=True)
    embedding = VectorField(dimensions=768, null=True, blank=True)

    class Meta:
        db_table = 'system_fields'

    def __str__(self):
        return f"{self.entity.name}.{self.name}" if self.entity and self.name else f"Field {self.id}"


class SystemRelationship(models.Model):
    """System relationships table - stores relationships between entities."""
    source_entity = models.ForeignKey(
        SystemEntity,
        on_delete=models.CASCADE,
        related_name='outgoing_relationships',
        null=True,
        blank=True
    )
    target_entity = models.ForeignKey(
        SystemEntity,
        on_delete=models.CASCADE,
        related_name='incoming_relationships',
        null=True,
        blank=True
    )
    relation_type = models.TextField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)

    class Meta:
        db_table = 'system_relationships'

    def __str__(self):
        source = self.source_entity.name if self.source_entity else "?"
        target = self.target_entity.name if self.target_entity else "?"
        return f"{source} -> {target} ({self.relation_type})"


class Hadith(models.Model):
    """Hadith model - stores hadith texts with narrators and embeddings."""
    id = models.BigAutoField(primary_key=True)
    hadith_number = models.PositiveIntegerField(null=True, blank=True)
    narrator = models.CharField(max_length=255, null=True, blank=True)
    narrator_arabic = models.TextField(null=True, blank=True)
    hadith_text_turkish = models.TextField(null=True, blank=True)
    hadith_text_arabic = models.TextField(null=True, blank=True)
    language = models.CharField(max_length=10, default='tr-ar')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'hadiths'
        ordering = ['hadith_number']
        indexes = [
            models.Index(fields=['hadith_number'], name='idx_hadith_number'),
            models.Index(fields=['language'], name='idx_hadith_language'),
        ]

    def __str__(self):
        return f"Hadith #{self.hadith_number} - {self.narrator or 'Unknown'}"


class HadithEmbedding(models.Model):
    """Hadith embeddings - stores vector embeddings for Turkish and Arabic texts."""
    id = models.BigAutoField(primary_key=True)
    hadith = models.OneToOneField(
        Hadith,
        on_delete=models.CASCADE,
        related_name='embeddings'
    )
    hadith_number = models.PositiveIntegerField(null=True, blank=True)
    embedding_turkish = VectorField(dimensions=768, null=True, blank=True)
    embedding_arabic = VectorField(dimensions=768, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'hadith_embeddings'
        indexes = [
            models.Index(fields=['hadith_number'], name='idx_hadith_emb_number'),
            # HNSW indexes for fast cosine similarity search
            HnswIndex(
                name='idx_emb_turkish_hnsw',
                fields=['embedding_turkish'],
                m=16,               # Max connections per layer (default 16)
                ef_construction=64, # Size of dynamic candidate list for construction (default 64)
                opclasses=['vector_cosine_ops'],  # Cosine distance
            ),
            HnswIndex(
                name='idx_emb_arabic_hnsw',
                fields=['embedding_arabic'],
                m=16,
                ef_construction=64,
                opclasses=['vector_cosine_ops'],
            ),
        ]

    def __str__(self):
        return f"Embeddings for Hadith #{self.hadith_number}"


class HadithSource(models.Model):
    """Hadith source references - stores source book and reference information."""
    hadith = models.ForeignKey(
        Hadith,
        on_delete=models.CASCADE,
        related_name='sources'
    )
    name = models.CharField(max_length=255)
    reference = models.CharField(max_length=255)

    class Meta:
        db_table = 'hadith_sources'

    def __str__(self):
        return f"{self.name} - {self.reference}"


class HadithChunkEmbedding(models.Model):
    """
    Hadith chunk embeddings - stores vector embeddings for individual text chunks.
    Each hadith can have multiple chunks for better semantic search precision.
    """
    id = models.BigAutoField(primary_key=True)
    hadith = models.ForeignKey(
        Hadith,
        on_delete=models.CASCADE,
        related_name='chunk_embeddings'
    )
    hadith_number = models.PositiveIntegerField(null=True, blank=True)
    chunk_index = models.PositiveIntegerField()
    chunk_text = models.TextField()
    language = models.CharField(max_length=10, choices=[
        ('turkish', 'Turkish'),
        ('arabic', 'Arabic'),
    ])
    embedding = VectorField(dimensions=768)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'hadith_chunk_embeddings'
        ordering = ['hadith_number', 'language', 'chunk_index']
        indexes = [
            models.Index(fields=['hadith_number'], name='idx_chunk_hadith_number'),
            models.Index(fields=['language'], name='idx_chunk_language'),
            models.Index(fields=['hadith_number', 'language'], name='idx_chunk_hadith_lang'),
            HnswIndex(
                name='idx_chunk_emb_hnsw',
                fields=['embedding'],
                m=16,
                ef_construction=64,
                opclasses=['vector_cosine_ops'],
            ),
        ]

    def __str__(self):
        return f"Hadith #{self.hadith_number} - {self.language} chunk {self.chunk_index}"
