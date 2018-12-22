from unittest import TestCase
from medacy.tools import DataLoader
import shutil, tempfile, os
from medacy.pipeline_components import MetaMap

class TestDataManager(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp() #set up test directory

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def tearDown(self):

        txt = os.path.join(self.test_dir, 'test.txt')
        if os.path.isfile(txt):
            os.remove(txt)

        ann = os.path.join(self.test_dir, 'test.ann')
        if os.path.isfile(ann):
            os.remove(ann)

    def test_init(self):
        """
        Tests initialization of DataManager
        :return:
        """
        with open(os.path.join(self.test_dir, 'test.txt'), 'w') as f:
            f.write("Contents of test file")

        with open(os.path.join(self.test_dir, 'test.ann'), 'w') as f:
            f.write("Example Ann File")

        self.assertIsInstance( DataLoader(self.test_dir) , DataLoader)

    def test_get_files(self):
        """
        Tests a text file and annotations being added and the correct length of all_files being set.
        :return:
        """
        with open(os.path.join(self.test_dir, 'test.txt'), 'w') as f:
            f.write("Contents of test file")

        with open(os.path.join(self.test_dir, 'test.ann'), 'w') as f:
            f.write("Example Ann File")

        data_loader = DataLoader(self.test_dir)
        self.assertEqual( len(data_loader.get_files()) , 1)


#Metamapping does not work in unit tests due to issues with streaming between processes - maybe address later
    # def test_metamapping(self):
    #     with open(os.path.join(self.test_dir, 'test.txt'), 'w') as f:
    #         f.write("Contents of test file")
    #
    #     with open(os.path.join(self.test_dir, 'test.ann'), 'w') as f:
    #         f.write("Example Ann File")
    #
    #     loader = DataLoader(self.test_dir)
    #
    #     metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap",
    #                       cache_output=False)
    #
    #     loader.metamap(metamap)
    #
    #
    #     """
    #     Metamap output cannot be tested through this wrapper due to how unit tests work (they override input/output
    #     streams which the wrapper relies on for calling the Metamap server). Instead we test that metamap works
    #     correctly and throws no errors - it will however output a null body for the metamapped contents in this test.
    #     """
    #
    #     with open(self.test_dir + "/metamapped/test.metamapped", 'r') as f:
    #         self.assertEqual("{\"metamap\": null}", f.read())




