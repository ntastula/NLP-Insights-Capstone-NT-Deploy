from django.core.management.base import BaseCommand
from myapp.utils.session_utils import SessionManager

class Command(BaseCommand):
    help = 'Clean up old sessions and associated data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=7,
            help='Delete sessions older than this many days (default: 7)',
        )

    def handle(self, *args, **options):
        days_old = options['days']
        deleted_count = SessionManager.cleanup_old_sessions(days_old)
        self.stdout.write(
            self.style.SUCCESS(f'Successfully deleted {deleted_count} old sessions')
        )