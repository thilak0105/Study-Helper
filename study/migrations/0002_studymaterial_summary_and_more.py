# Generated by Django 5.1.4 on 2025-01-11 06:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("study", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="studymaterial",
            name="summary",
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="studymaterial",
            name="summary_generated_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
