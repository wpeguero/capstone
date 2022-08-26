import unittest
import pipeline

class TestPipeline(unittest.TestCase):
    test_name = "TCIA_TCGA-PRAD_08-09-2016-v3"
    def test_data_output(self):
        self.assertEqual(type(pipeline.get_data(name)), list)
