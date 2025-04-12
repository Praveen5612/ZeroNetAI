class GPIOMock:
    BOARD = "BOARD"
    OUT = "OUT"
    IN = "IN"
    HIGH = 1
    LOW = 0
    
    @staticmethod
    def setmode(mode):
        pass
    
    @staticmethod
    def setup(pin, mode):
        pass
    
    @staticmethod
    def output(pin, value):
        pass
    
    @staticmethod
    def cleanup():
        pass

class SpiDevMock:
    def __init__(self):
        pass
    
    def open(self, bus, device):
        pass