from django.db import migrations, models
import django.db.models.deletion
from pgvector.django import VectorField, VectorExtension


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        # Enable pgvector extension
        VectorExtension(),

        # Create ApiEndpoint model (api_endpoints table)
        migrations.CreateModel(
            name='ApiEndpoint',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('path', models.TextField(blank=True, null=True)),
                ('method', models.TextField(blank=True, null=True)),
                ('summary', models.TextField(blank=True, null=True)),
                ('parameters', models.JSONField(blank=True, null=True)),
                ('responses', models.JSONField(blank=True, null=True)),
                ('embedding', VectorField(blank=True, dimensions=768, null=True)),
                ('last_updated', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'api_endpoints',
            },
        ),

        # Create DocsChunk model (docs_chunks table)
        migrations.CreateModel(
            name='DocsChunk',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField(blank=True, null=True)),
                ('metadata', models.JSONField(blank=True, null=True)),
                ('embedding', VectorField(blank=True, dimensions=768, null=True)),
                ('last_updated', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'docs_chunks',
            },
        ),

        # Create SystemEntity model (system_entities table)
        migrations.CreateModel(
            name='SystemEntity',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.TextField(blank=True, null=True)),
                ('description', models.TextField(blank=True, null=True)),
                ('metadata', models.JSONField(blank=True, null=True)),
                ('embedding', VectorField(blank=True, dimensions=768, null=True)),
                ('last_updated', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name_plural': 'System entities',
                'db_table': 'system_entities',
            },
        ),

        # Create SystemField model (system_fields table)
        migrations.CreateModel(
            name='SystemField',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.TextField(blank=True, null=True)),
                ('type', models.TextField(blank=True, null=True)),
                ('options', models.JSONField(blank=True, null=True)),
                ('embedding', VectorField(blank=True, dimensions=768, null=True)),
                ('entity', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='fields', to='chatApp.systementity')),
            ],
            options={
                'db_table': 'system_fields',
            },
        ),

        # Create SystemRelationship model (system_relationships table)
        migrations.CreateModel(
            name='SystemRelationship',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('relation_type', models.TextField(blank=True, null=True)),
                ('description', models.TextField(blank=True, null=True)),
                ('source_entity', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='outgoing_relationships', to='chatApp.systementity')),
                ('target_entity', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='incoming_relationships', to='chatApp.systementity')),
            ],
            options={
                'db_table': 'system_relationships',
            },
        ),

        # Add indexes for DocsChunk
        migrations.AddIndex(
            model_name='docschunk',
            index=models.Index(fields=['last_updated'], name='idx_docs_chunks_last_updated'),
        ),

        # Add indexes for SystemEntity
        migrations.AddIndex(
            model_name='systementity',
            index=models.Index(fields=['last_updated'], name='idx_system_entities_last_upd'),
        ),

        # Add index for api_endpoints last_updated
        migrations.AddIndex(
            model_name='apiendpoint',
            index=models.Index(fields=['last_updated'], name='idx_api_endpoints_last_updat'),
        ),

        # GIN index on docs_chunks.metadata
        migrations.RunSQL(
            sql="CREATE INDEX IF NOT EXISTS idx_docs_chunks_metadata ON docs_chunks USING GIN (metadata);",
            reverse_sql="DROP INDEX IF EXISTS idx_docs_chunks_metadata;",
        ),

        # Create the update_updated_at_column function
        migrations.RunSQL(
            sql="""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.last_updated = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """,
            reverse_sql="DROP FUNCTION IF EXISTS update_updated_at_column();",
        ),

        # Create trigger for docs_chunks
        migrations.RunSQL(
            sql="""
                DROP TRIGGER IF EXISTS update_docs_chunks_last_updated ON docs_chunks;
                CREATE TRIGGER update_docs_chunks_last_updated
                    BEFORE UPDATE ON docs_chunks
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            """,
            reverse_sql="DROP TRIGGER IF EXISTS update_docs_chunks_last_updated ON docs_chunks;",
        ),

        # Create trigger for api_endpoints
        migrations.RunSQL(
            sql="""
                DROP TRIGGER IF EXISTS update_api_endpoints_last_updated ON api_endpoints;
                CREATE TRIGGER update_api_endpoints_last_updated
                    BEFORE UPDATE ON api_endpoints
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            """,
            reverse_sql="DROP TRIGGER IF EXISTS update_api_endpoints_last_updated ON api_endpoints;",
        ),

        # Create trigger for system_entities
        migrations.RunSQL(
            sql="""
                DROP TRIGGER IF EXISTS update_system_entities_last_updated ON system_entities;
                CREATE TRIGGER update_system_entities_last_updated
                    BEFORE UPDATE ON system_entities
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            """,
            reverse_sql="DROP TRIGGER IF EXISTS update_system_entities_last_updated ON system_entities;",
        ),

        # Create knowledge_index VIEW
        migrations.RunSQL(
            sql="""
                CREATE OR REPLACE VIEW knowledge_index AS
                SELECT id, 'api' AS source_type, path AS title, summary AS content, embedding
                FROM api_endpoints
                UNION ALL
                SELECT id, 'docs' AS source_type, text AS title, metadata::text AS content, embedding
                FROM docs_chunks
                UNION ALL
                SELECT id, 'entity' AS source_type, name AS title, description AS content, embedding
                FROM system_entities;
            """,
            reverse_sql="DROP VIEW IF EXISTS knowledge_index;",
        ),
    ]
