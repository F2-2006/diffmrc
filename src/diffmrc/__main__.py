"""``python -m diffmrc`` runs the command line."""
import sys

from .cli.main import main

if __name__ == '__main__':
    sys.exit(main())
