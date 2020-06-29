import unittest

from src.main import handle_plots_folder, handle_dataset_folder, get_dataset_files


class TestFilesDirectories(unittest.TestCase):
    def test_plots_directory_not_exists(self):
        """
         Test that it creates a directory
         """
        result = handle_plots_folder()
        self.assertEqual(result, 1)

    def test_plots_directory_exists(self):
        """
         Test if a directory exists
         """
        result = handle_plots_folder()
        self.assertEqual(result, 0)

    def test_data_directory_exists_and_contains_file(self):
        """
         Test if a directory exists and contains any files
         """
        result = handle_dataset_folder()
        self.assertEqual(result, 0)

    def test_list_of_files(self):
        """
         Test if could retrieve dataset direcotry files
         """
        result = get_dataset_files()
        print(result)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
