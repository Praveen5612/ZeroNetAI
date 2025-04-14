from flask import Flask, render_template, request, jsonify, url_for
from flask_socketio import SocketIO, emit
import logging
import os

logger = logging.getLogger(__name__)

class WebInterface:
    def __init__(self, ai_instance):
        self.ai = ai_instance
        self.app = Flask(__name__, 
                        static_folder='static',
                        template_folder='templates')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
    def initialize(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')
            
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            message = request.json.get('message')
            response = self.ai.model_manager.generate_response(message)
            # Store the conversation
            try:
                self.ai.message_store.store_conversation(
                    user_input=message,
                    ai_response=response,
                    source="web",
                    session_id=request.remote_addr
                )
                logger.info(f"Stored conversation from {request.remote_addr}")
            except Exception as e:
                logger.error(f"Failed to store conversation: {e}")
            
            return jsonify({'response': response})
            
    def run(self, host='0.0.0.0', port=None):
        """Run the web interface"""
        port = int(os.environ.get('PORT', 5000))
        if os.environ.get('ENVIRONMENT') == 'production':
            # Use Gunicorn in production (Render)
            import gunicorn
            options = {
                'bind': f'{host}:{port}',
                'workers': 4,
                'worker_class': 'eventlet',
                'timeout': 120
            }
            gunicorn.run(self.app, **options)
        else:
            # Use Flask's built-in server for development
            self.app.run(host=host, port=port, debug=True)
        self.socketio.run(self.app, host=host, port=port, debug=True)