"""
Basic tests to verify project setup
"""

import sys
import os
import unittest

# Add src to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestProjectSetup(unittest.TestCase):
    """Test that the project is set up correctly"""
    
    def test_imports_work(self):
        """Test that we can import our packages"""
        try:
            import src.models
            import src.services
            import src.data
            import src.utils
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_config_exists(self):
        """Test that configuration file exists"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.py')
        self.assertTrue(os.path.exists(config_path))
    
    def test_data_directories_exist(self):
        """Test that data directories exist"""
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.assertTrue(os.path.exists(data_dir))

if __name__ == '__main__':
    unittest.main() 