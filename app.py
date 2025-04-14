from src.web.app import WebInterface
from src.llm.model_manager import ModelManager
from src.database.message_store import MessageStore

class AI:
    def __init__(self):
        self.model_manager = ModelManager()
        self.message_store = MessageStore()

    def initialize(self):
        self.model_manager.load_model()

if __name__ == "__main__":
    ai = AI()
    ai.initialize()
    web = WebInterface(ai)
    web.initialize()
    web.run()