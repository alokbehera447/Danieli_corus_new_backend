"""
App configuration for the planner app.
"""

from django.apps import AppConfig


class PlannerConfig(AppConfig):
    """Configuration for the planner app."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'planner'
    verbose_name = 'Cutting Optimization Planner'
