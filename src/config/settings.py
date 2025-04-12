import os
from pathlib import Path
from dotenv import load_dotenv

class Settings:
    def __init__(self):
        load_dotenv()
        
        # Base paths
        self.base_dir = Path("c:/Documents/projects/AI")
        self.model_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        
        # Device settings
        self.device_id = os.getenv("DEVICE_ID", "zero_net_default")  # Added device_id
        
        # Model settings
        self.model_name = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        # LoRa settings
        self.lora_frequency = float(os.getenv("LORA_FREQUENCY", "915.0"))
        self.lora_bandwidth = int(os.getenv("LORA_BANDWIDTH", "125"))
        self.lora_tx_power = int(os.getenv("LORA_TX_POWER", "17"))
        
        # Bluetooth settings
        self.bluetooth_name = os.getenv("BLUETOOTH_NAME", "ZeroNetAI")
        self.bluetooth_channel = int(os.getenv("BLUETOOTH_CHANNEL", "1"))