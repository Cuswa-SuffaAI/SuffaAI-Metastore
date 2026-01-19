from django.db import models
from pgvector.django import VectorField


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
