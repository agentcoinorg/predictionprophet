import logging

class BaseLogger:
    def __init__(self):
        pass
    
    def debug(self, msg, *args, **kwargs):
        logging.debug(msg, *args)
    
    def info(self, msg, *args, **kwargs):
        logging.info(msg, *args)
    
    def warning(self, msg, *args, **kwargs):
        logging.warning(msg, *args)
    
    def error(self, msg, *args, **kwargs):
        logging.error(msg, *args)
    
    def critical(self, msg, *args, **kwargs):
        logging.critical(msg, *args)
