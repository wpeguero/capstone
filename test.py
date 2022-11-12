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

    test_name = "TCIA_TCGA-PRAD_08-09-2016-v3"
    def test_data_output(self):
        self.assertEqual(type(pipeline.get_data(name)), list)
