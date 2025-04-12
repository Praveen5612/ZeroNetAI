import logging
import asyncio
import json
from config.settings import Settings
from llm.model_manager import ModelManager
from communication.lora_manager import LoRaManager
from communication.bluetooth_manager import BluetoothManager
from database.message_store import MessageStore
from web.app import WebInterface
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZeroNetAI:
    def __init__(self):
        self.settings = Settings()
        self.model_manager = ModelManager()
        self.lora_manager = LoRaManager()
        self.bluetooth_manager = BluetoothManager()
        self.message_store = MessageStore()
        self.web_interface = WebInterface(self)
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing ZeroNetAI...")
        self.model_manager.load_model()
        self.lora_manager.initialize()
        await self.bluetooth_manager.initialize()
        self.message_store.initialize()
        self.web_interface.initialize()
    
    async def run(self):
        """Main application loop"""
        await self.initialize()
        logger.info("ZeroNetAI is running...")
        
        # Start web interface in a separate thread
        web_thread = threading.Thread(target=self.web_interface.run)
        web_thread.daemon = True
        web_thread.start()
        
        while True:
            # Handle LoRa messages
            if msg := self.lora_manager.receive_message():
                response = self.model_manager.generate_response(msg)
                self.lora_manager.send_message(response)
            
            # Handle Bluetooth messages
            if msg := await self.bluetooth_manager.receive_message():
                response = self.model_manager.generate_response(msg)
                await self.bluetooth_manager.send_message(response)
            
            await asyncio.sleep(0.1)  # Prevent CPU overuse

if __name__ == "__main__":
    app = ZeroNetAI()
    asyncio.run(app.run())