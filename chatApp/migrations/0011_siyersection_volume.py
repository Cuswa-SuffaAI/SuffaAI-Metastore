from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatApp', '0010_fetvaquestion_fetvachunkembedding'),
    ]

    operations = [
        migrations.AddField(
            model_name='siyersection',
            name='volume',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
