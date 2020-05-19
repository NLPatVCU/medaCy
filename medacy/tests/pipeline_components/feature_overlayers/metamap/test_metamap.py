import os
import shutil
import tempfile
import unittest

from medacy.data.dataset import Dataset
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
from medacy.tests.pipeline_components.feature_overlayers.metamap import have_metamap, reason, metamap_path
from medacy.tests.sample_data import sample_dataset


class TestMetaMap(unittest.TestCase):
    """Unit tests for medacy.pipeline_components.feature_overlayers.metamap.metamap.MetaMap"""

    @classmethod
    def setUpClass(cls) -> None:
        if not have_metamap:
            return
        cls.metamap = MetaMap(metamap_path)
        cls.metamap.activate()

        # Create an unmetamapped copy of the sample dataset
        cls.temp_dataset_dir = tempfile.mkdtemp()
        for df in sample_dataset:
            shutil.copyfile(df.txt_path, os.path.join(cls.temp_dataset_dir, df.file_name + '.txt'))
            shutil.copyfile(df.ann_path, os.path.join(cls.temp_dataset_dir, df.file_name + '.ann'))

        cls.dataset = Dataset(cls.temp_dataset_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        if not have_metamap:
            return
        cls.metamap.deactivate()
        shutil.rmtree(cls.temp_dataset_dir)

    @unittest.skipUnless(have_metamap, reason)
    def test_map_file(self):
        """Tests that mapping a file has expected output"""
        txt_file = sample_dataset.data_files[0].txt_path
        mm_output = self.metamap.map_file(txt_file)
        self.assertIsInstance(mm_output, dict)
        self.assertIn('metamap', mm_output.keys())

    @unittest.skipUnless(have_metamap, reason)
    def test_map_text(self):
        """Tests that mapping a string has expected output"""
        sample_str = """One week after ovariectomy, animals were treated with estrogen, using groups of three animals per treatment condition. In the first PSSD study, 15 rats (five groups of three rats) were injected subcutaneously with either 17  -E2 (60  g/kg; 12 rats) or the sesame oil vehicle (200  L; three rats). Nine of the 12 estradiol-treated animals were treated simultaneously with increasing doses (40, 120, and 400  g/kg) of BPA (> 99% purity; Sigma-Aldrich, St. Louis, MO, USA). In the second PSSD experiment, 12 rats were injected subcutaneously (three rats per treatment) with 17  -E2 (45  g/kg), BPA (300  g/kg), a combination of 17  -E2 (45  g/kg) plus BPA (300  g/kg), or the sesame oil vehicle (200  L) alone. Thirty minutes after injection, animals were sacrificed under deep ether anesthesia by transcardial perfusion of heparinized saline followed by a fixative containing 4% paraformaldehyde and 0.1% glutaraldehyde in 0.1 M phosphate buffer (pH 7.35). The brains were removed and postfixed overnight in the same fixative without glutaraldehyde. The hippocampi were then dissected out, and 100  m vibratome sections were cut perpendicular to the longitudinal axis of the hippocampus. The approximately 90 vibratome sections were divided into 10 subgroups using systematic random sampling and were flat-embedded in Araldite (Electron Microscopy Sciences, Fort Washington, PA, USA). 
To correct for processing-induced changes in the volume of the tissue, we calculated a correction factor assuming that the treatments did not alter the total number of pyramidal cells. In all hippocampi, we examined six or seven disector pairs (pairs of adjacent 2- m semi-thin sections mounted on slides and stained with toluidine blue). We calculated a pyramidal cell density value (D) using the formula D = N/sT, where N is the mean disector score across all sampling windows, T is the thickness of the sections (2  m), and s is the length of the window. Based on these values, a dimensionless volume correction factor kv was introduced: kv = D/D1, where D1 is the mean cell density across the groups of hippocampi (Rusakov et al. 1997)."""
        mm_output = self.metamap.map_text(sample_str)
        self.assertIsInstance(mm_output, dict)
        self.assertIn('metamap', mm_output.keys())

    @unittest.skipUnless(have_metamap, reason)
    def test_map_dataset(self):
        """Tests metamapping a Dataset object"""
        self.assertFalse(self.dataset.is_metamapped())
        self.metamap.metamap_dataset(self.dataset, 2)
        self.assertTrue(self.dataset.is_metamapped())
        for df in self.dataset:
            self.assertTrue(os.path.isfile(df.metamapped_path))


if __name__ == '__main__':
    unittest.main()
