import logging

class BaseLogger:
    def __init__(self) -> None:
        pass
    
    def debug(self, msg: str) -> None:
        logging.debug(msg)
    
    def info(self, msg: str) -> None:
        logging.info(msg)
    
    def warning(self, msg: str) -> None:
        logging.warning(msg)
    
    def error(self, msg: str) -> None:
        logging.error(msg)
    
    def critical(self, msg: str) -> None:
        logging.critical(msg)
