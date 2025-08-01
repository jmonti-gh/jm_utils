"""
jm_logger
Advanced log management module with automatic rotation and directory creation.

Features:
- Automatic directory creation
- Log rotation by maximum count
- Flexible log level configuration
- Lazy initialization (doesn't create files until used)
- Support for local paths and network shares
- Customizable application name and title

Usage:
    from log_manager import LogManager
    
    # Configure the logger
    logger = LogManager.setup(
        log_path="C:/logs/my_app",
        max_logs=5,
        level="INFO",
        app_name="MyApplication",
        title="Application v1.0 - Production Environment"
    )
    
    # Use the logger
    logger.info("Information message")
    logger.error("Error message")
"""

__version__ = "0.1.0"
__description__ = "Utilities I use frequently - Several modules"
__author__ = "Jorge Monti"
__email__ = "jorgitomonti@gmail.com"
__license__ = "MIT"
__status__ = "Development"
__python_requires__ = ">=3.11"
__last_modified__ = "2025-06-15"


import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


class LogManager:
    """
    Log manager with automatic rotation and directory creation.
    """
    
    _instance = None
    _logger = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def setup(cls, 
              log_path: str,
              max_logs: int = 10,
              level: str = "INFO",
              app_name: str = "app",
              title: str = "",
              log_format: Optional[str] = None,
              date_format: Optional[str] = None) -> logging.Logger:
        """
        Configure the logging system.
        
        Args:
            log_path: Path where to store logs (disk, folder or network share)
            max_logs: Maximum number of log files to maintain
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            app_name: Application name for log filename (replaces 'app' prefix)
            title: Title/header information to be logged at the beginning
            log_format: Custom format for log messages
            date_format: Custom format for dates
            
        Returns:
            logging.Logger: Configured logger
            
        Raises:
            ValueError: If parameters are invalid
            OSError: If directory cannot be created or path is inaccessible
        """
        instance = cls()
        
        if instance._initialized:
            return instance._logger
        
        # Validate parameters
        if not log_path or not isinstance(log_path, str):
            raise ValueError("log_path must be a valid path")
        
        if not isinstance(max_logs, int) or max_logs < 1:
            raise ValueError("max_logs must be an integer greater than 0")
        
        # Validate log level
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
        
        # Validate app_name
        if not isinstance(app_name, str) or not app_name.strip():
            raise ValueError("app_name must be a non-empty string")
        
        # Clean app_name for filename usage (remove invalid characters)
        app_name_clean = "".join(c for c in app_name if c.isalnum() or c in "._-").strip()
        if not app_name_clean:
            app_name_clean = "app"
        
        # Configure paths and parameters
        instance.log_path = Path(log_path)
        instance.max_logs = max_logs
        instance.level = level.upper()
        instance.app_name = app_name_clean
        instance.title = title.strip() if isinstance(title, str) else ""
        
        # Create log directory if it doesn't exist
        instance._create_log_directory()
        
        # Configure logger
        instance._setup_logger(log_format, date_format)
        
        # Configure rotation handler (lazy - only when first log is written)
        instance._setup_rotation_handler()
        
        instance._initialized = True
        return instance._logger
    
    def _create_log_directory(self):
        """Create log directory if it doesn't exist."""
        try:
            self.log_path.mkdir(parents=True, exist_ok=True)
            
            # Verify write permissions
            test_file = self.log_path / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
            except (PermissionError, OSError) as e:
                raise OSError(f"No write permissions in: {self.log_path}") from e
                
        except OSError as e:
            raise OSError(f"Error creating log directory {self.log_path}: {e}") from e
    
    def _setup_logger(self, log_format: Optional[str], date_format: Optional[str]):
        """Configure the main logger."""
        
        # Default format
        if log_format is None:
            log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        if date_format is None:
            date_format = '%Y-%m-%d %H:%M:%S'
        
        # Create logger
        self._logger = logging.getLogger('jm_logger')
        self._logger.setLevel(getattr(logging, self.level))
        
        # Avoid duplicating handlers if they already exist
        if self._logger.handlers:
            self._logger.handlers.clear()
        
        # Configure formatter
        formatter = logging.Formatter(log_format, date_format)
        
        # Console handler (optional)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.level))
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        
        # FileHandler will be created when first log is written
        self._file_handler = None
        self._formatter = formatter
    
    def _setup_rotation_handler(self):
        """Configure log rotation handling (lazy initialization)."""
        # Override logging methods to implement lazy initialization
        original_handle = self._logger.handle
        
        def lazy_handle(record):
            if self._file_handler is None:
                self._create_file_handler()
            return original_handle(record)
        
        self._logger.handle = lazy_handle
    
    def _create_file_handler(self):
        """Create file handler when needed for the first time."""
        if self._file_handler is not None:
            return
        
        # Rotate existing logs before creating a new one
        self._rotate_logs()
        
        # Create filename with timestamp using custom app_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.app_name}_{timestamp}.log"
        log_file_path = self.log_path / log_filename
        
        # Create file handler
        self._file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        self._file_handler.setLevel(getattr(logging, self.level))
        self._file_handler.setFormatter(self._formatter)
        
        # Add to logger
        self._logger.addHandler(self._file_handler)
        
        # Write initial log entries
        self._write_log_header(log_file_path)
    
    def _write_log_header(self, log_file_path: Path):
        """Write initial header information to the log file."""
        separator = "=" * 60
        
        # Session start header
        self._logger.info(separator)
        self._logger.info("LOGGING SESSION STARTED")
        self._logger.info(separator)
        
        # Custom title if provided
        if self.title:
            self._logger.info(f"TITLE: {self.title}")
            self._logger.info("-" * 60)
        
        # Log configuration information
        self._logger.info(f"Log file: {log_file_path}")
        self._logger.info(f"Application: {self.app_name}")
        self._logger.info(f"Log level: {self.level}")
        self._logger.info(f"Maximum logs: {self.max_logs}")
        self._logger.info(f"Session start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.title:
            self._logger.info(f"Custom title: {self.title}")
        
        self._logger.info(separator)
    
    def _rotate_logs(self):
        """Delete old logs if they exceed the maximum allowed."""
        try:
            # Search for existing log files with the custom app_name
            log_pattern = f"{self.app_name}_*.log"
            log_files = list(self.log_path.glob(log_pattern))
            
            # Sort by modification date (most recent first)
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Delete excess files
            files_to_delete = log_files[self.max_logs-1:]  # -1 because we're creating a new one
            
            for log_file in files_to_delete:
                try:
                    log_file.unlink()
                    print(f"Log deleted: {log_file}")
                except OSError as e:
                    print(f"Warning: Could not delete {log_file}: {e}")
                    
        except Exception as e:
            print(f"Warning: Error during log rotation: {e}")
    
    @classmethod
    def get_logger(cls) -> Optional[logging.Logger]:
        """
        Get the configured logger.
        
        Returns:
            logging.Logger or None if not configured
        """
        instance = cls()
        return instance._logger if instance._initialized else None
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the logger has been initialized."""
        instance = cls()
        return instance._initialized
    
    @classmethod
    def get_current_config(cls) -> dict:
        """
        Get current logger configuration.
        
        Returns:
            dict: Current configuration parameters
        """
        instance = cls()
        if not instance._initialized:
            return {}
        
        return {
            'log_path': str(instance.log_path),
            'max_logs': instance.max_logs,
            'level': instance.level,
            'app_name': instance.app_name,
            'title': instance.title,
            'initialized': instance._initialized
        }
    
    @classmethod
    def reset(cls):
        """Reset logger configuration."""
        instance = cls()
        if instance._logger:
            for handler in instance._logger.handlers[:]:
                handler.close()
                instance._logger.removeHandler(handler)
        
        instance._logger = None
        instance._initialized = False
        instance._file_handler = None


# Convenience functions for direct usage
def setup_logging(log_path: str, 
                 max_logs: int = 10, 
                 level: str = "INFO",
                 app_name: str = "app",
                 title: str = "",
                 log_format: Optional[str] = None,
                 date_format: Optional[str] = None) -> logging.Logger:
    """
    Convenience function to configure logging.
    
    Args:
        log_path: Path where to store logs
        max_logs: Maximum number of log files
        level: Logging level
        app_name: Application name for log filename
        title: Title/header information for log file
        log_format: Custom format for messages
        date_format: Custom format for dates
        
    Returns:
        logging.Logger: Configured logger
    """
    return LogManager.setup(log_path, max_logs, level, app_name, title, log_format, date_format)


def get_logger() -> Optional[logging.Logger]:
    """
    Get the configured logger.
    
    Returns:
        logging.Logger or None if not configured
    """
    return LogManager.get_logger()


def get_config() -> dict:
    """
    Get current logger configuration.
    
    Returns:
        dict: Current configuration
    """
    return LogManager.get_current_config()


# Usage example
if __name__ == "__main__":
    LOG_PATH = './logs_1'

    # Configuration example
    logger = setup_logging(
        log_path=LOG_PATH,
        max_logs=5,
        level="DEBUG",
        app_name="PasoSIMU",
        title="PasoSIMU System v2.1 - Production Environment - Server: PROD-01"
    )
    
    # Usage examples
    logger.debug("Debug message")
    logger.info("Application started successfully")
    logger.warning("This is a warning")
    logger.error("Example error")
    logger.critical("Critical error example")
    
    print(f"Logs written successfully. Check the '{LOG_PATH}' folder")
    print(f"Current configuration: {get_config()}")