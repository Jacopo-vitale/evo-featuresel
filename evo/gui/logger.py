import logging
from PySide6.QtCore import QObject, Signal

class QtLoggingHandler(logging.Handler, QObject):
    """
    Custom logging handler that emits a PySide6 signal for every log record.
    Allows for real-time log display in a QPlainTextEdit.
    """
    log_signal = Signal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        # Match the style of the original CLI logger
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)
