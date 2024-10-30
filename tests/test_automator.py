import unittest
import os
import sys

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, mock_open
from automation import create_readme, identify_dependencies, generate_code_structure

class TestAutomation(unittest.TestCase):

    @patch('os.walk')
    @patch('builtins.open', new_callable=mock_open, read_data='import os\nimport re\nfrom collections import Counter\n')
    def test_identify_dependencies(self, mock_open, mock_os_walk):
        mock_os_walk.return_value = [
            ('/some/path', ('subdir',), ('file1.py', 'file2.py')),
            ('/some/path/subdir', (), ('file3.py',)),
        ]
        expected_dependencies = ['Counter']
        dependencies = identify_dependencies('/some/path')
        self.assertEqual(dependencies, expected_dependencies)

    @patch('os.walk')
    def test_generate_code_structure(self, mock_os_walk):
        mock_os_walk.return_value = [
            ('/some/path', ('subdir',), ('file1.py', 'file2.py')),
            ('/some/path/subdir', (), ('file3.py',)),
        ]
        expected_structure = "- file1.py: Main module or functionality\n- file2.py: Main module or functionality\n- subdir/file3.py: Main module or functionality"
        structure = generate_code_structure('/some/path')
        self.assertEqual(structure, expected_structure)

    @patch('os.walk')
    @patch('builtins.open', new_callable=mock_open)
    @patch('automation.generate_readme_summary')
    def test_create_readme(self, mock_generate_readme_summary, mock_open, mock_os_walk):
        mock_os_walk.return_value = [
            ('/some/path', ('subdir',), ('file1.py', 'file2.py')),
            ('/some/path/subdir', (), ('file3.py',)),
        ]
        mock_generate_readme_summary.return_value = "README content"
        
        create_readme('/some/path')
        
        mock_open.assert_called_once_with('/some/path/README_suggestion.md', 'w')
        mock_open().write.assert_called_once_with("README content")

if __name__ == '__main__':
    unittest.main()