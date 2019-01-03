"""
Manages training data
"""
import os, json, logging, math
from .data_file import DataFile
import multiprocessing, warnings
from joblib import Parallel, delayed


class DataLoader:

    def __init__(self, data_directory, raw_text_file_extension="txt", annotation_file_extension="ann",  limit=None):
        """
        Manages directory of training data along with other relevant files.
        A directory must consist of at-least A.txt , A.ann file pairs. Specifically, the DataLoader interfaces
        with metamap to do pre-metamapping of files and associating the pre-mapped files with ann, txt pairs. By
        default DataLoader will create a sub-directory of a directory of text files which contains associated metamapped
        files for each text file in the parent directory.

        :param data_directory: Directory containing training data consisting of raw and annotated text pairs
        :param raw_text_file_extension: the file extension of the text files
        :param annotation_file_extension: the file extension of the annotation files
        :param limit: number of documents to utilize (defaults to all in directory)
        """
        self.data_directory = data_directory
        self.raw_text_file_extension = raw_text_file_extension
        self.all_files = []
        files = os.listdir(data_directory)

        #start by filtering all raw_text files, both training and prediction directories will have these
        raw_text_files = sorted([file for file in files if file.endswith(raw_text_file_extension)])

        assert raw_text_files is not None, "No raw text files exist in directory: %s" % self.data_directory

        if limit is not None and 0<limit and limit <= len(raw_text_files):
            raw_text_files = raw_text_files[0:limit]

        #required ann files for this to be a training directory
        ann_files = [file.replace(".%s" % raw_text_file_extension, ".%s" % annotation_file_extension) for file in raw_text_files]

        #only a training directory if every text file has a corresponding ann_file
        is_training_directory = all([os.path.isfile(os.path.join(data_directory, ann_file)) for ann_file in ann_files])


        #set all file attributes except metamap_path as it is optional.
        for file in raw_text_files:
            file_name = file[:-len(raw_text_file_extension)-1]
            raw_text_path = os.path.join(data_directory, file)

            if is_training_directory:
                ann_path = os.path.join(data_directory, file.replace(".%s" % raw_text_file_extension, ".%s" % annotation_file_extension))
            else:
                ann_path = None
            self.all_files.append(DataFile(file_name, raw_text_path, ann_path) )

        if self.is_metamapped():
            for file in self.all_files:
                file.metamapped_path = os.path.join(self.data_directory + "/metamapped",
                                                    file.raw_path.split(os.path.sep)[-1].replace(
                                                        ".%s" % self.raw_text_file_extension, ".metamapped"))
        logging.debug("Loaded %i files for %s", len(raw_text_files), "training" if is_training_directory else "prediction")

    def get_files(self):
        """
        Retrieves an organized list containing the files processed.
        :return: a list of DataFile objects
        """
        return self.all_files

    def _parallel_run(self, files, i):
        """
        Facilitates Metamapping in parallel by forking off processes to Metamap each file individually
        :param files: an array of file paths to the file to map
        :param i: index in the array used to determine the file that the called process will be responsible for mapping
        :return:
        """
        file = files[i].split(os.path.sep)[-1]
        file_path = files[i]
        logging.info("Attempting to Metamap: %s", file_path)
        mapped_file_location = os.path.join(self.data_directory+"/metamapped", file.replace(self.raw_text_file_extension, "metamapped"))
        if not os.path.isfile(mapped_file_location):
            mapped_file = open(mapped_file_location, 'w')
            max_prune_depth = 30  # this is the maximum prune depth metamap utilizes when concept mapping

            metamap_dict = None
            while metamap_dict is None or metamap_dict['metamap'] is None: #while current prune depth causes out of memory on document
                try:
                    metamap_dict = self.metamap.map_file(file_path, max_prune_depth=max_prune_depth) #attempt to metamap
                    if metamap_dict['metamap'] is not None: #if successful
                        break
                    max_prune_depth = int(math.e ** (math.log(max_prune_depth) - .5)) #decrease prune depth by an order of magnitude
                except BaseException as e:
                    if max_prune_depth <= 0: # Lowest prune depth reached, abort MetaMapping
                        logging.warning("Can not Metamap file %s, lowest prune depth reached", file_path)
                        metamap_dict = ''
                        break
                    else:
                        metamap_dict = None
                        max_prune_depth = int(math.e ** (math.log(max_prune_depth) - .5)) #decrease prune depth by an order of magnitude
                        logging.warning("Error Metamapping: %s with exception %s", file_path, str(e))

            mapped_file.write(json.dumps(metamap_dict))
            logging.info("Successfully Metamapped: %s", file_path)
            logging.info("Successfully Metamapped: %s" % file_path)



    def is_metamapped(self):
        return os.path.isdir(self.data_directory+"/metamapped/")

    def metamap(self, metamap, n_jobs = multiprocessing.cpu_count()-1):
        """
        Metamaps training data and places it in a new sub_directory 'metamapped'
        :param n_jobs: number metamap processes to fork. Default is to use all cores.
        Will add metamap file path to the returned paths
        """
        self.metamap = metamap
        metamapped_files_directory = self.data_directory+"/metamapped/"

        if not self.is_metamapped():
            os.makedirs(metamapped_files_directory)
            # logging.warning("Metamap directory already exists, please delete if you are attempting to re-map. Exiting mapping.")
            # return

        already_metamapped = [file[:file.find('.')] for file in os.listdir(metamapped_files_directory)]
        files = [file.raw_path for file in self.all_files if not file.file_name in already_metamapped]
        logging.info("Number of files to MetaMap: %i" % len(files))

        Parallel(n_jobs=n_jobs)(delayed(self._parallel_run)(files, i) for i in range(len(files)))

        for file in self.all_files:
            file.metamapped_path = os.path.join(self.data_directory+"/metamapped", file.raw_path.split(os.path.sep)[-1].replace(".%s" % self.raw_text_file_extension, ".metamapped"))









