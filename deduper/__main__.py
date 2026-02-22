from .app import create_app, socketio

if __name__ == '__main__':
    app = create_app()
    # Use socketio.run() instead of app.run() to enable WebSocket support.
    # This wraps the Flask dev server with Socket.IO's WSGI middleware.
    socketio.run(
        app,
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=True,
        use_reloader=True,
        allow_unsafe_werkzeug=True,  # Required for Flask-SocketIO with Werkzeug dev server
    )
