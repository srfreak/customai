class ScrizaException(Exception):
    """Base exception for Scriza platform"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class AuthenticationException(ScrizaException):
    """Exception raised for authentication errors"""
    pass

class AuthorizationException(ScrizaException):
    """Exception raised for authorization errors"""
    pass

class AgentNotFoundException(ScrizaException):
    """Exception raised when agent is not found"""
    pass

class StrategyNotFoundException(ScrizaException):
    """Exception raised when strategy is not found"""
    pass

class InvalidStrategyException(ScrizaException):
    """Exception raised when strategy is invalid"""
    pass

class MemoryOperationException(ScrizaException):
    """Exception raised for memory operation errors"""
    pass

class VoiceSynthesisException(ScrizaException):
    """Exception raised for voice synthesis errors"""
    pass

class TelephonyException(ScrizaException):
    """Exception raised for telephony errors"""
    pass
