import hashlib
import json
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

@dataclass
class MeshPacket:
    message_id: str
    source_id: str
    target_id: str
    content: str
    timestamp: float
    hop_count: int
    message_type: str
    
class MeshProtocol:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.seen_messages = set()
        self.max_hop_count = 5
        
    def create_packet(self, content: str, target_id: str = "broadcast") -> dict:
        message_id = hashlib.md5(f"{content}{datetime.now().timestamp()}".encode()).hexdigest()
        
        packet = MeshPacket(
            message_id=message_id,
            source_id=self.device_id,
            target_id=target_id,
            content=content,
            timestamp=datetime.now().timestamp(),
            hop_count=0,
            message_type="message"
        )
        
        return self._packet_to_dict(packet)
    
    def process_packet(self, packet_data: dict) -> Optional[dict]:
        packet = self._dict_to_packet(packet_data)
        
        if packet.message_id in self.seen_messages:
            return None
            
        self.seen_messages.add(packet.message_id)
        
        if packet.target_id != "broadcast" and packet.target_id != self.device_id:
            if packet.hop_count < self.max_hop_count:
                packet.hop_count += 1
                return self._packet_to_dict(packet)
            return None
            
        return self._packet_to_dict(packet)