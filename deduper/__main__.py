from .app import create_app
from .config import config

if __name__ == '__main__':
    app = create_app()
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=True,
        threaded=True  # Enable threading to prevent blocking on long-running requests
    ) 