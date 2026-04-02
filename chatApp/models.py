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
    """Hadith model - stores hadith texts with narrators and metadata."""
    id = models.BigAutoField(primary_key=True)
    hadith_number = models.PositiveIntegerField(null=True, blank=True)
    narrator = models.CharField(max_length=255, null=True, blank=True)
    narrator_arabic = models.TextField(null=True, blank=True)
    hadith_text_turkish = models.TextField(null=True, blank=True)
    hadith_text_arabic = models.TextField(null=True, blank=True)
    language = models.CharField(max_length=10, default='tr-ar')
    # Enriched metadata fields
    potential_questions = models.JSONField(null=True, blank=True)
    embedding_text = models.TextField(null=True, blank=True)
    priority_keywords = models.JSONField(null=True, blank=True)
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
    Hadith chunk embeddings - stores vector embeddings for embedding_text and potential_questions.
    Each hadith has:
    - 1 embedding_text chunk (summary for semantic search)
    - Multiple potential_question chunks (8-12 questions per hadith)
    """
    CHUNK_TYPE_CHOICES = [
        ('embedding_text', 'Embedding Text'),
        ('potential_question', 'Potential Question'),
    ]

    id = models.BigAutoField(primary_key=True)
    hadith = models.ForeignKey(
        Hadith,
        on_delete=models.CASCADE,
        related_name='chunk_embeddings'
    )
    hadith_number = models.PositiveIntegerField(null=True, blank=True)
    chunk_type = models.CharField(max_length=20, choices=CHUNK_TYPE_CHOICES, default='embedding_text')
    chunk_index = models.PositiveIntegerField(default=0)
    chunk_text = models.TextField()
    embedding = VectorField(dimensions=768)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'hadith_chunk_embeddings'
        ordering = ['hadith_number', 'chunk_type', 'chunk_index']
        indexes = [
            models.Index(fields=['hadith_number'], name='idx_chunk_hadith_number'),
            models.Index(fields=['chunk_type'], name='idx_chunk_type'),
            models.Index(fields=['hadith_number', 'chunk_type'], name='idx_chunk_hadith_type'),
            HnswIndex(
                name='idx_chunk_emb_hnsw',
                fields=['embedding'],
                m=16,
                ef_construction=64,
                opclasses=['vector_cosine_ops'],
            ),
        ]

    def __str__(self):
        return f"Hadith #{self.hadith_number} - {self.chunk_type} [{self.chunk_index}]"


# ============================================================
# SIYER MODELLERI
# ============================================================

class SiyerSection(models.Model):
    """Siyer bölüm modeli - PDF'den çıkarılan bölüm metinleri ve metadata."""
    id = models.BigAutoField(primary_key=True)
    section_id = models.CharField(max_length=50, unique=True)  # siyer_1b257eea63da
    text = models.TextField()
    pages = models.JSONField(null=True, blank=True)  # [12, 13, 14]
    volume = models.JSONField(null=True, blank=True)  # ["1-2. Cilt"]
    summary_short = models.TextField(null=True, blank=True)
    main_theme = models.CharField(max_length=255, null=True, blank=True)
    potential_questions = models.JSONField(null=True, blank=True)  # ["Soru 1?", ...]
    query_intents = models.JSONField(null=True, blank=True)  # ["anahtar1", ...]
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'siyer_sections'
        ordering = ['id']
        indexes = [
            models.Index(fields=['section_id'], name='idx_siyer_section_id'),
            models.Index(fields=['main_theme'], name='idx_siyer_main_theme'),
        ]

    def __str__(self):
        return f"{self.section_id} - {self.main_theme or 'N/A'}"


class SiyerChunkEmbedding(models.Model):
    """
    Siyer chunk embeddings - bölüm metinleri için vektör embedding'leri.
    Her bölüm için:
    - 1 summary chunk (kısa özet)
    - 10 potential_question chunk (sorular)
    - N query_intent chunk (anahtar kavramlar)
    """
    CHUNK_TYPE_CHOICES = [
        ('summary', 'Summary'),
        ('potential_question', 'Potential Question'),
        ('query_intent', 'Query Intent'),
    ]

    id = models.BigAutoField(primary_key=True)
    section = models.ForeignKey(
        SiyerSection,
        on_delete=models.CASCADE,
        related_name='chunk_embeddings'
    )
    section_code = models.CharField(max_length=50)  # siyer_1b257eea63da
    chunk_type = models.CharField(max_length=20, choices=CHUNK_TYPE_CHOICES)
    chunk_index = models.PositiveIntegerField(default=0)
    chunk_text = models.TextField()
    embedding = VectorField(dimensions=768)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'siyer_chunk_embeddings'
        ordering = ['section_code', 'chunk_type', 'chunk_index']
        indexes = [
            models.Index(fields=['section_code'], name='idx_siyer_chunk_sec_code'),
            models.Index(fields=['chunk_type'], name='idx_siyer_chunk_type'),
            models.Index(fields=['section_code', 'chunk_type'], name='idx_siyer_chunk_code_type'),
            HnswIndex(
                name='idx_siyer_chunk_emb_hnsw',
                fields=['embedding'],
                m=16,
                ef_construction=64,
                opclasses=['vector_cosine_ops'],
            ),
        ]

    def __str__(self):
        return f"{self.section_code} - {self.chunk_type} [{self.chunk_index}]"


# ============================================================
# FETVA MODELLERI
# ============================================================

class FetvaQuestion(models.Model):
    """Fetva questions - stores question, answer and metadata."""
    id = models.BigAutoField(primary_key=True)
    fetva_id = models.CharField(max_length=50, unique=True)  # e0ca3e868553
    question = models.TextField()
    answer = models.TextField()
    page = models.PositiveIntegerField(null=True, blank=True)
    subject = models.CharField(max_length=100, null=True, blank=True)  # itikat, fıkıh, etc.
    paraphrase_questions = models.JSONField(null=True, blank=True)  # ["Question 1?", ...]
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'fetva_questions'
        ordering = ['id']
        indexes = [
            models.Index(fields=['fetva_id'], name='idx_fetva_id'),
            models.Index(fields=['subject'], name='idx_fetva_subject'),
        ]

    def __str__(self):
        return f"{self.fetva_id} - {self.question[:60]}"


class FetvaChunkEmbedding(models.Model):
    """
    Fetva chunk embeddings - vector embeddings for questions.
    Per fetva:
    - 1 question chunk (original question)
    - N paraphrase_question chunks (paraphrased questions)
    """
    CHUNK_TYPE_CHOICES = [
        ('question', 'Question'),
        ('paraphrase_question', 'Paraphrase Question'),
    ]

    id = models.BigAutoField(primary_key=True)
    fetva = models.ForeignKey(
        FetvaQuestion,
        on_delete=models.CASCADE,
        related_name='chunk_embeddings'
    )
    fetva_code = models.CharField(max_length=50)  # e0ca3e868553
    chunk_type = models.CharField(max_length=20, choices=CHUNK_TYPE_CHOICES)
    chunk_index = models.PositiveIntegerField(default=0)
    chunk_text = models.TextField()
    embedding = VectorField(dimensions=768)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'fetva_chunk_embeddings'
        ordering = ['fetva_code', 'chunk_type', 'chunk_index']
        indexes = [
            models.Index(fields=['fetva_code'], name='idx_fetva_chunk_fetva_code'),
            models.Index(fields=['chunk_type'], name='idx_fetva_chunk_type'),
            models.Index(fields=['fetva_code', 'chunk_type'], name='idx_fetva_chunk_code_type'),
            HnswIndex(
                name='idx_fetva_chunk_emb_hnsw',
                fields=['embedding'],
                m=16,
                ef_construction=64,
                opclasses=['vector_cosine_ops'],
            ),
        ]

    def __str__(self):
        return f"{self.fetva_code} - {self.chunk_type} [{self.chunk_index}]"
