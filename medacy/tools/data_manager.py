"""
Manages training data
"""
import os, json, logging

import multiprocessing, warnings
from joblib import Parallel, delayed

class DataFile():
    def __init__(self, file_name, raw_text_fp, ann_path):
        self.file_name = file_name
        self.raw_path = raw_text_fp
        self.ann_path = ann_path
        self.metamapped_path = None
    def __repr__(self):
        return self.file_name
    def __str__(self):
        return self.file_name


class DataLoader():

    def __init__(self, data_directory, raw_text_file_extension="txt"):
        """
        Manages directory of training data along with other relevant files.
        A directory must consist of at-least A.txt , A.ann file pairs
        :param data_directory: Directory containing training data consisting of raw and annotated text pairs
        """
        self.data_directory = data_directory
        self.raw_text_file_extension = raw_text_file_extension
        self.all_files = []
        files = os.listdir(data_directory)
        assert any(File.endswith(".%s" % raw_text_file_extension) for File in os.listdir(data_directory)), "Directory contains no %s files" % raw_text_file_extension

        is_training_directory = any(File.endswith(".ann") for File in os.listdir(data_directory))

        #set all file attributes except metamap_path as it is optional.
        for file in files:
            if file.endswith(".%s" % raw_text_file_extension):
                file_name = file.replace(".%s" % raw_text_file_extension, "")
                raw_text_path = os.path.join(data_directory, file)

                if is_training_directory:
                    ann_path = os.path.join(data_directory, file.replace(".%s" % raw_text_file_extension, ".ann"))

                    if not os.path.isfile(ann_path):
                        raise FileNotFoundError("Could not find file: %s" % ann_path)
                else:
                    ann_path = None
                self.all_files.append(DataFile(file_name, raw_text_path, ann_path))

        if self.is_metamapped():
            for file in self.all_files:
                file.metamapped_path = os.path.join(self.data_directory + "/metamapped",
                                                    file.raw_path.split(os.path.sep)[-1].replace(
                                                        ".%s" % self.raw_text_file_extension, ".metamapped"))

    def get_files(self):
        return self.all_files

    def _parallel_run(self, files, i):

        file = files[i].split(os.path.sep)[-1]
        file_path = files[i]
        logging.info("Attempting to Metamap: %s", file_path)
        print("Attempting to Metamap: %s" % file_path)
        mapped_file_location = os.path.join(self.data_directory+"/metamapped", file.replace(self.raw_text_file_extension, "metamapped"))
        if not os.path.isfile(mapped_file_location):
            mapped_file = open(mapped_file_location, 'w')
            try:
                mapped_file.write(json.dumps(self.metamap.map_file(file_path)))
                logging.info("Successfully Metamapped: %s", file_path)
                print("Successfully Metamapped: %s" % file_path)
            except Exception as e:
                logging.warning("Error Metamapping: %s with exception %s", file_path, str(e))
                mapped_file.write(str(e))


    def is_metamapped(self):
        return os.path.isdir(self.data_directory+"/metamapped/")


    def metamap(self, metamap, num_cores = multiprocessing.cpu_count()):
        """
        Metamaps training data and places it in a new sub_directory 'metamapped'
        :param num_cores: number of cores to spawn metamap processes on. Default is to use all cores.
        Will add metamap file path to the returned paths
        """
        self.metamap = metamap
        metamapped_files_directory = self.data_directory+"/metamapped/"

        if self.is_metamapped():
            logging.warning("Metamap directory already exists, please delete if you are attempting to re-map. Exiting mapping.")
            return
        os.makedirs(metamapped_files_directory)

        files = [file.raw_path for file in self.all_files]


        Parallel(n_jobs=num_cores)(delayed(self._parallel_run)(files, i) for i in range(len(files)))

        for file in self.all_files:
            file.metamapped_path = os.path.join(self.data_directory+"/metamapped", file.raw_path.split(os.path.sep)[-1].replace(".%s" % self.raw_text_file_extension, ".metamapped"))









