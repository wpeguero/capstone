"""Set of tests for observing the functionality of the algoriths."""
import unittest
import pipeline

class TestPipeline(unittest.TestCase):
    """Test the data pipeline.

    Class made to test whether the pipeline is able to
    appropriately load data for training or testing. This
    will also be used to examine whether the data is
    extracted and/or transformed without encountering an
    unknown error.
    """
    def test_data_extraction(self):
        self.assertEqual()

if __name__ == "__main__":
    unittest.main()
