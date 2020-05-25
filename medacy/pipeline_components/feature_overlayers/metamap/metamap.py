import json
import logging
import multiprocessing
import os
import subprocess
import tempfile
import warnings
import math

import xmltodict
from joblib import Parallel, delayed

from medacy.tools.unicode_to_ascii import UNICODE_TO_ASCII


class MetaMap:
    """A python wrapper for MetaMap that includes built in caching of MetaMap output."""

    def __init__(self, metamap_path, cache_output=False, cache_directory=None, convert_ascii=True, args=""):
        """
        :param metamap_path: The location of the MetaMap executable.
            (ex. /home/programs/metamap/2016/public_mm/bin/metamap)
        :param cache_output: Whether to cache output as it run through metamap, will by default store in a
            temp directory tmp/medacy*/
        :param cache_directory: alternatively, specify a directory to cache metamapped files to
        """

        # Set cache directory to tmp directory, creating if not exists
        if cache_output and cache_directory is None:
            tmp = tempfile.gettempdir()

            files = [filename for filename in os.listdir(tmp) if filename.startswith("medacy")]
            if files:
                cache_directory = os.path.join(tmp, files[0])
            else:
                tmp_dir = tempfile.mkdtemp(prefix="medacy")
                cache_directory = os.path.join(tmp, tmp_dir)

        self.cache_directory = cache_directory
        self.metamap_path = metamap_path
        self.convert_ascii = convert_ascii
        self.args = args
        # Set path to the program that enables metamapping
        self._program_name = os.path.join(os.path.dirname(self.metamap_path), 'skrmedpostctl')
        self.recent_file = None
        self.metamap_dict = {}

    def activate(self):
        """Activates MetaMap for metamapping files or strings"""
        subprocess.call([self._program_name, 'start'])

    def __enter__(self):
        """Activates MetaMap for metamapping files or strings"""
        self.activate()
        return self

    def deactivate(self):
        """Deactivates MetaMap"""
        subprocess.call([self._program_name, 'stop'])

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Deactivates MetaMap"""
        self.deactivate()

    def map_file(self, file_to_map, max_prune_depth=10):
        """
        Maps a given document from a file_path and returns a formatted dict
        :param file_to_map: the path of the file to be metamapped
        :param max_prune_depth: See metamap specs about pruning depth; defaults to 10; set to larger for better results.
        :return: a dictionary of MetaMap data
        """
        self.recent_file = file_to_map

        if self.cache_directory is not None:  # Look up file if exists, otherwise continue metamapping
            cached_file_path = os.path.join(
                self.cache_directory,
                os.path.splitext(os.path.basename(file_to_map))[0] + ".metamapped"
            )

            if os.path.exists(cached_file_path):
                logging.debug(cached_file_path)
                return self.load(cached_file_path)

        with open(file_to_map, 'r') as f:
            contents = f.read()

        metamap_dict = self._run_metamap('--XMLf --blanklines 0 --silent --prune %i %s' % (max_prune_depth, self.args), contents)

        if self.cache_directory is not None:
            with open(cached_file_path, 'w') as mapped_file:
                try:
                    mapped_file.write(json.dumps(metamap_dict))
                except Exception as e:
                    logging.error(str(e))

        return metamap_dict

    def map_text(self, text, max_prune_depth=10):
        """
        Runs MetaMap over str input
        :param text: A string to run MetaMap over
        :param max_prune_depth: defaults to 10
        :return: a MetaMap dict
        """
        self.metamap_dict = self._run_metamap('--XMLf --blanklines 0 --silent --prune %i' % max_prune_depth, text)
        return self.metamap_dict

    @staticmethod
    def load(file_to_load):
        with open(file_to_load, 'rb') as f:
            return json.load(f)

    def _run_metamap(self, args, document):
        """
        Runs metamap through bash and feeds in appropriate arguments
        :param args: arguments to feed into metamap
        :param document: the raw text to be metamapped
        :return:
        """
        if self.convert_ascii:
            document, ascii_diff = self._convert_to_ascii(document)

        bash_command = 'bash %s %s' % (self.metamap_path, args)
        process = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate(input=bytes(document, 'UTF-8'))
        output = str(output.decode('utf-8'))

        xml = ""
        lines = output.split('\n')

        # Lines at index 1 and 2 are a header for the XML output
        for line in lines[1:3]:
            xml += line + '\n'

        # The beginning of the metamap-specific XML is this tag
        xml += "<metamap>\n"

        for line in output.split("\n")[3:]:
            if not all(item in line for item in ['DOCTYPE', 'xml']):
                xml += line + '\n'

        xml += "</metamap>"  # surround in single root tag - hacky.

        if output is None:
            raise Exception("An error occured while using metamap: %s" % error)

        metamap_dict = xmltodict.parse(xml)

        if self.convert_ascii:
            document, metamap_dict = self._restore_from_ascii(document, ascii_diff, metamap_dict)

        return metamap_dict

    def _item_generator(self, json_input, lookup_key):
        if isinstance(json_input, dict):
            for k, v in json_input.items():
                if k == lookup_key:
                    yield v
                else:
                    yield from self._item_generator(v, lookup_key)
        elif isinstance(json_input, list):
            for item in json_input:
                yield from self._item_generator(item, lookup_key)

    def extract_mapped_terms(self, metamap_dict):
        """
        Extracts an array of term dictionaries from metamap_dict
        :param metamap_dict: A dictionary containing the metamap output
        :return: an array of mapped_terms
        """
        if metamap_dict['metamap'] is None:
            warnings.warn("Metamap output is none for a file in the pipeline. Exiting.")
            return

        all_terms = []

        for term in self._item_generator(metamap_dict, 'Candidate'):
            if isinstance(term, dict):
                all_terms.append(term)
            if isinstance(term, list):
                all_terms = all_terms + term

        return all_terms

    def mapped_terms_to_spacy_ann(self, mapped_terms, entity_label=None):
        """
        Transforms an array of mapped_terms in a spacy annotation object. Label for each annotation
        defaults to first semantic type in semantic_type array
        :param mapped_terms: an array of mapped terms
        :param entity_label: the label to assign to each annotation, defaults to first semantic type of mapped_term
        :return: a annotation formatted to spacy's specifications
        """

        annotations = []

        for term in mapped_terms:
            for span in self.get_span_by_term(term):  # if a single entity corresonds to a disjunct span
                entity_start, entity_end = span
                if entity_label is None:
                    annotations.append((entity_start, entity_end, self.get_semantic_types_by_term(term)[0]))
                else:
                    annotations.append((entity_start, entity_end, entity_label))

        return annotations

    def get_term_by_semantic_type(self, mapped_terms, include=[], exclude=None):
        """
        Returns metamapped utterances that all contain a given set of semantic types found in include
        :param mapped_terms: An array of candidate dictionaries
        :return: the dictionaries that contain a term with all the semantic types in semantic_types
        """

        if exclude is not None:
            intersection = set(include) & exclude
            if intersection:
                raise Exception("Include and exclude overlap with the following semantic types: " + ", ".join(intersection))
        matches = []

        for term in mapped_terms:

            found_types = []

            if int(term['SemTypes']['@Count']) == 0:
                continue
            if int(term['SemTypes']['@Count']) == 1:
                found_types.append(term['SemTypes']['SemType'])
            if int(term['SemTypes']['@Count']) > 1:
                found_types = term['SemTypes']['SemType']
            if exclude is not None and set(exclude) <= set(found_types):
                continue

            if set(include) <= set(found_types):
                matches.append(term)

        return matches

    def get_span_by_term(self, term):
        """
        Takes a given utterance dictionary (term) and extracts out the character indices of the utterance
        :param term: The full dictionary corresponding to a metamap term
        :return: the span of the referenced term in the document
        """
        if isinstance(term['ConceptPIs']['ConceptPI'], list):
            spans = []
            for span in term['ConceptPIs']['ConceptPI']:
                start = int(span['StartPos'])
                length = int(span['Length'])
                spans.append((start, start + length))
            return spans
        else:
            start = int(term['ConceptPIs']['ConceptPI']['StartPos'])
            length = int(term['ConceptPIs']['ConceptPI']['Length'])
            return [(start, start + length)]

    def get_semantic_types_by_term(self, term):
        """
        Returns an array of the semantic types of a given term
        :param term:
        :return:
        """
        if int(term['SemTypes']['@Count']) == 1:
            return [term['SemTypes']['SemType']]

        return term['SemTypes']['SemType']

    def __call__(self, file_path):
        """
        Metamaps a file and returns an array of mapped terms from the file
        :param file_path:
        :return: array of mapped_terms
        """
        metamap_dict = self.map_file(file_path)

        return self.extract_mapped_terms(metamap_dict)

    def _convert_to_ascii(self, text):
        """Takes in a text string and converts it to ASCII,
        keeping track of each character change

        The changes are recorded in a list of objects, each object
        detailing the original non-ASCII character and the starting
        index and length of the replacement in the new string (keys
        ``original``, ``start``, and ``length``, respectively).

        Args:
            text (string): The text to be converted
        
        Returns:
            tuple: tuple containing:

                **text** (*string*): The converted text

                **diff** (*list*): Record of all ASCII conversions
        """
        diff = list()
        offset = 0
        for i, char in enumerate(text):
            if ord(char) >= 128: #non-ascii
                if char in UNICODE_TO_ASCII and UNICODE_TO_ASCII[char] is not char:
                    ascii = UNICODE_TO_ASCII[char]
                    text = text[:i+offset] + ascii + text[i+1+offset:]
                    diff.append({
                        'start': i+offset,
                        'length': len(ascii),
                        'original': char
                    })
                    offset += len(ascii) - len(char)
                else:
                    ascii = '?'
                    text = text[:i + offset] + ascii + text[i + 1 + offset:]
                    diff.append({
                        'start': i + offset,
                        'length': len(ascii),
                        'original': char
                    })
                    offset += len(ascii) - len(char)
        return text, diff

    def _restore_from_ascii(self, text, diff, metamap_dict):
        """Takes in non-ascii text and the list of changes made to it from the `convert()` function,
        as well as a dictionary of metamap taggings, converts the text back to its original state
        and updates the character spans in the metamap dict to match

        Arguments:
            text (string): Output of ``_convert_to_ascii()``
            diff (list): Output of ``_convert_to_ascii()``
            metamap_dict (dict): Dictionary of metamap information obtained from ``text``
        
        Returns:
            tuple: tuple containing:

                **text** (*string*): The input with all of the changes listed in ``diff`` reversed
                **metamap_dict** (*dict*): The input with all of its character spans updated to reflect the changes to ``text``
        """
        offset = 0
        for conv in diff: # Go through each recorded change to undo it & update metamap character spans accordingly
            conv_start = conv['start'] + offset
            conv_end = conv_start + conv['length']-1 # Ending index of converted span, INCLUSIVE

            # Undo the change to the text (restore ascii characters)
            text = text[:conv_start] + conv['original'] + text[conv_end+1:]
            delta = len(conv['original']) - conv['length']
            offset += delta

            # Check each metamap entry and update its character spans to reflect this change
            # > I'm so sorry it looks like this, but because of the way we convert the xml
            # > into a dict, there are some levels in the hierarchy that are usually a list
            # > but turn into an object if they only contain one element, so that needs
            # > to be checked for at every step or it crashes the whole program
            if type(metamap_dict['metamap']['MMOs']['MMO']['Utterances']['Utterance']) is not list:
                metamap_dict['metamap']['MMOs']['MMO']['Utterances']['Utterance'] = [metamap_dict['metamap']['MMOs']['MMO']['Utterances']['Utterance']]

            for utterance in metamap_dict['metamap']['MMOs']['MMO']['Utterances']['Utterance']:
                if int(utterance['Phrases']['@Count']) == 0:  # Ensure this level contains something
                    continue
                if type(utterance['Phrases']['Phrase']) is not list:  # Make sure this entry is a list
                    utterance['Phrases']['Phrase'] = [utterance['Phrases']['Phrase']]

                for phrase in utterance['Phrases']['Phrase']:
                    if int(phrase['Mappings']['@Count']) == 0:  # Ensure this level contains something
                        continue
                    if type(phrase['Mappings']['Mapping']) is not list:  # Make sure this entry is a list
                        phrase['Mappings']['Mapping'] = [phrase['Mappings']['Mapping']]

                    for mapping in phrase['Mappings']['Mapping']:
                        if int(mapping['MappingCandidates']['@Total']) == 0:  # Ensure this level contains something
                            continue
                        if type(mapping['MappingCandidates']['Candidate']) is not list:  # Make sure this entry is a list
                            mapping['MappingCandidates']['Candidate'] = [mapping['MappingCandidates']['Candidate']]

                        # HERE'S THE IMPORTANT PART -----------------------------------------
                        # Just accept it as iterating through every entry in the metamap_dict
                        for candidate in mapping['MappingCandidates']['Candidate']:
                            if int(candidate['ConceptPIs']['@Count']) == 0:  # Ensure this level contains something
                                continue
                            if type(candidate['ConceptPIs']['ConceptPI']) is not list:  # Make sure this entry is a list
                                candidate['ConceptPIs']['ConceptPI'] = [candidate['ConceptPIs']['ConceptPI']]

                            candidate['MatchedWords']['MatchedWord'] = []
                            for conceptpi in candidate['ConceptPIs']['ConceptPI']:
                                match_start = int(conceptpi['StartPos'])
                                match_length = int(conceptpi['Length'])
                                match_end = match_start + match_length-1

                                if match_start == conv_start and match_end == conv_end:  # If match is equal to conversion (a [conversion] and some text)
                                    # print("Perfect match")
                                    match_length += delta
                                elif match_start < conv_start and match_end < conv_end:  # If match intersects conversion on left ([a con]version and some text)
                                    # print("Left intersect")
                                    match_length += delta + conv_start
                                elif conv_start < match_start and conv_end < match_end:  # If match intersects conversion on right (a conver[sion and som]e text)
                                    # print("Right intersect ")
                                    if conv_end + delta < match_start:
                                        match_start = conv_end + delta + 1
                                        match_length = match_end - conv_end
                                    else:
                                        match_length += delta
                                elif conv_end < match_start:  # If match is totally to the right of the conversion (a conversion and a [match])
                                    # print("Full right")
                                    match_start += delta
                                else:  # If match is totally to right of conversion, no action needed (a [match] and a conversion)
                                    # print("Full left")
                                    pass

                                # Update metamap entry with new indices
                                candidate['MatchedWords']['MatchedWord'].append(text[match_start:match_end+1])
                                conceptpi['StartPos'] = str(match_start)
                                conceptpi['Length'] = str(match_length)
        return text, metamap_dict

    def metamap_dataset(self, dataset, n_jobs=multiprocessing.cpu_count() - 1, retry_possible_corruptions=True):
        """
         Metamaps the files registered by a Dataset. Attempts to Metamap utilizing a max prune depth of 30, but on
         failure retries with lower max prune depth. A lower prune depth roughly equates to decreased MetaMap performance.
         More information can be found in the MetaMap documentation.

         :param dataset: the Dataset to MetaMap.
         :param n_jobs: the number of processes to spawn when metamapping. Defaults to one less core than available on your machine.
         :param retry_possible_corruptions: Re-Metamap's files that are detected as being possibly corrupt. Set to False for more control over what gets Metamapped or if you are having bugs with Metamapping. (default: True)
         :return: None
         """

        if dataset.is_metamapped():
            logging.info(f"The following Dataset has already been metamapped: {repr(dataset)}")
            return

        mm_dir = dataset.data_directory / "metamapped"

        # Make MetaMap directory if it doesn't exist.
        if not os.path.isdir(mm_dir):
            os.makedirs(mm_dir)
            dataset.metamapped_files_directory = mm_dir

        # A file that is below 200 bytes is likely corrupted output from MetaMap, these should be retried.
        if retry_possible_corruptions:
            # Do not metamap files that are already metamapped and above 200 bytes in size
            already_metamapped = [file[:file.find('.')] for file in os.listdir(mm_dir)
                                  if os.path.getsize(os.path.join(mm_dir, file)) > 200]
        else:
            # Do not metamap files that are already metamapped
            already_metamapped = [file[:file.find('.')] for file in os.listdir(mm_dir)]

        files_to_metamap = [data_file for data_file in dataset if data_file.file_name not in already_metamapped]

        logging.info(f"Number of files to MetaMap: {len(files_to_metamap)}")

        Parallel(n_jobs=n_jobs)(delayed(self._parallel_metamap)(file, mm_dir) for file in files_to_metamap)

        if not dataset.is_metamapped():
            raise RuntimeError(f"MetaMapping {dataset} was unsuccessful")

        for data_file in dataset:
            data_file.metamapped_path = os.path.join(
                mm_dir,
                data_file.file_name + ".metamapped"
            )

    def _parallel_metamap(self, data_file, mm_dir):
        """
        Facilitates metamapping in parallel by forking off processes to MetaMap each file individually.

        :param data_file: a DataFile to metamap
        :return: None
        """
        file_name = data_file.file_name
        file_path = data_file.txt_path
        logging.info("Attempting to Metamap: %s", file_path)
        mapped_file_location = os.path.join(mm_dir, file_name + ".metamapped")

        with open(mapped_file_location, 'w') as mapped_file:
            max_prune_depth = 30  # this is the maximum prune depth metamap utilizes when concept mapping

            metamap_dict = None
            # while current prune depth causes out of memory on document
            while metamap_dict is None or metamap_dict['metamap'] is None:
                if max_prune_depth <= 0:
                    logging.critical("Failed to to metamap after multiple attempts: %s", file_path)
                    return
                try:
                    metamap_dict = self.map_file(file_path, max_prune_depth=max_prune_depth)  # attempt to metamap
                    if metamap_dict['metamap'] is not None:  # if successful
                        break
                    # Decrease prune depth by an order of magnitude
                    max_prune_depth = int(math.e ** (math.log(max_prune_depth) - .5))
                except BaseException as e:
                    metamap_dict = None
                    # Decrease prune depth by an order of magnitude
                    max_prune_depth = int(math.e ** (math.log(max_prune_depth) - .5))
                    logging.warning(f"Error Metamapping: {file_path} after raising {type(e).__name__}: {str(e)}")

            mapped_file.write(json.dumps(metamap_dict))
            logging.info("Successfully Metamapped: %s", file_path)
