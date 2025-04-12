import logging
import asyncio
from bleak import BleakClient, BleakScanner
from typing import Optional

logger = logging.getLogger(__name__)

class BluetoothManager:
    def __init__(self):
        self.device = None
        self.connected = False
        
    async def initialize(self):
        """Initialize Bluetooth hardware"""
        logger.info("Initializing Bluetooth communication...")
        try:
            devices = await BleakScanner.discover()
            for d in devices:
                if d.name and "ZeroNetAI" in d.name:
                    self.device = d
                    break
            logger.info("Bluetooth initialized successfully")
        except Exception as e:
            logger.error(f"Bluetooth initialization failed: {e}")
            raise
    
    async def send_message(self, message: str) -> bool:
        """Send message via Bluetooth"""
        if not self.device:
            logger.error("Bluetooth device not initialized")
            return False
            
        try:
            async with BleakClient(self.device.address) as client:
                await client.write_gatt_char(self.CHAR_UUID, message.encode())
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_message(self) -> Optional[str]:
        """Receive message via Bluetooth"""
        if not self.device:
            return None
            
        try:
            async with BleakClient(self.device.address) as client:
                data = await client.read_gatt_char(self.CHAR_UUID)
                return data.decode()
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None