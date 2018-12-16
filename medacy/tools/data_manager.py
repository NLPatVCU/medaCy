"""
Manages training data
"""
import os, json, logging, math

import multiprocessing, warnings
from joblib import Parallel, delayed


class DataFile:
    def __init__(self, file_name, raw_text_fp, ann_path):
        self.file_name = file_name
        self.raw_path = raw_text_fp
        self.ann_path = ann_path
        self.metamapped_path = None

    def __repr__(self):
        return self.file_name

    def __str__(self):
        return self.file_name


class DataLoader:

    def __init__(self, data_directory, raw_text_file_extension="txt", limit=None):
        """
        Manages directory of training data along with other relevant files.
        A directory must consist of at-least A.txt , A.ann file pairs. Specifically, the DataLoader interfaces
        with metamap to do pre-metamapping of files and associating the pre-mapped files with ann, txt pairs. By
        default DataLoader will create a sub-directory of a directory of text files which contains associated metamapped
        files for each text file in the parent directory.

        :param data_directory: Directory containing training data consisting of raw and annotated text pairs
        :param raw_text_file_extension: extension of annotated text files
        :param limit: number of documents to utilize (defaults to all in directory)
        """
        self.data_directory = data_directory
        self.raw_text_file_extension = raw_text_file_extension
        self.all_files = []
        files = os.listdir(data_directory)
        if limit is not None and 0 < limit and limit <= len(files):
            files = files[:limit]
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
        """
        Returns a list of all files processed by DataLoader
        :return:
        """
        return self.all_files

    def _parallel_run(self, files, i):

        file = files[i].split(os.path.sep)[-1]
        file_path = files[i]
        logging.info("Attempting to Metamap: %s", file_path)
        mapped_file_location = os.path.join(self.data_directory+"/metamapped", file.replace(self.raw_text_file_extension, "metamapped"))
        if not os.path.isfile(mapped_file_location):
            mapped_file = open(mapped_file_location, 'w')
            try:
                max_prune_depth = 25 #this is the maximum prune depth metamap utilizes when concept mapping
                metamap_dict = self.metamap.map_file(file_path, max_prune_depth=max_prune_depth)
                while metamap_dict['metamap'] is None: #while current prune depth causes out of memory on document
                    max_prune_depth = int(math.e ** (math.log(max_prune_depth) - 1)) #decrease prune depth by an order of magnitude
                    metamap_dict = self.metamap.map_file(file_path, max_prune_depth=max_prune_depth)#and try again
                mapped_file.write(json.dumps(metamap_dict))
                logging.info("Successfully Metamapped: %s", file_path)
                logging.info("Successfully Metamapped: %s" % file_path)
            except Exception as e:
                logging.warning("Error Metamapping: %s with exception %s", file_path, str(e))
                mapped_file.write(str(e))


    def is_metamapped(self):
        return os.path.isdir(self.data_directory+"/metamapped/")


    def metamap(self, metamap, num_cores = multiprocessing.cpu_count()-1):
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









