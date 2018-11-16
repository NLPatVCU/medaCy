"""
A utility class to Metamap medical text documents.
Metamap a file  and utilize it the output or manipulate stored metamap output

"""
import subprocess
import xmltodict
import json
import tempfile, os, warnings


class MetaMap:

    def __init__(self, metamap_path=None, cache_output = False, cache_directory = None):
        """

        A python wrapper for metamap that includes built in caching of metamap output.

        :param cache_output: Whether to cache output as it run through metamap, will by default store in a
                             temp directory tmp/medacy*/
        :param cache_directory: alternatively, specify a directory to cache metamapped files to
        :param metamap_path: The location of the metamap executable.
                            (ex. /home/share/programs/metamap/2016/public_mm/bin/metamap)
        """
        if metamap_path is None:
            raise ValueError("metamap_path is not set. Insure Metamap is running and a path to the metamap executable is being given (ex. metamap/2016/public_mm/bin/metamap)")

        if cache_output:
            if cache_directory is None: #set cache directory to tmp directory, creating if not exists
                tmp = tempfile.gettempdir()
                files = [filename for filename in os.listdir(tmp) if filename.startswith("medacy")]

                if files:
                    cache_directory = os.path.join(tmp,files[0])
                else:
                    tmp_dir = tempfile.mkdtemp(prefix="medacy")
                    cache_directory = os.path.join(tmp, tmp_dir)

        self.cache_directory = cache_directory
        self.metamap_path = metamap_path

    def map_file(self, file_to_map, max_prune_depth=10):
        """
        Maps a given document from a file_path and returns a formatted dict
        :param file_to_map: the path of the file that will be metamapped
        :param max_prune_depth: set to larger if you know what you are doing. See metamap specs about pruning depth.
        :return:
        """
        self.recent_file = file_to_map

        file = open(file_to_map, 'r')
        if not file:
            raise FileNotFoundError("Error opening file while attempting to map: %s" % file_to_map)

        if self.cache_directory is not None: #look up file if exists, otherwise continue metamapping
            cached_file_path = os.path.join(
                self.cache_directory,
                os.path.splitext(os.path.basename(file_to_map))[0] + ".metamapped"
            )

            if os.path.exists(cached_file_path):
                print(cached_file_path)
                return self.load(cached_file_path)


        contents = file.read()
        metamap_dict = self._run_metamap('--XMLf --blanklines 0 --silent --prune %i' % max_prune_depth, contents)

        if self.cache_directory is not None:
            with open(cached_file_path, 'w') as mapped_file:
                try:
                    #print("Writing to", os.path.join(self.cache_directory, file_name))
                    mapped_file.write(json.dumps(metamap_dict))
                except Exception as e:
                    mapped_file.write(str(e))

        return metamap_dict

    def map_text(self, text, max_prune_depth=10):
        #TODO add caching here as in map_file
        self.metamap_dict = self._run_metamap('--XMLf --blanklines 0 --silent --prune %i' % max_prune_depth, text)
        return self.metamap_dict

    def load(self, file_to_load):
        with open(file_to_load, 'r') as f:
            return json.load(f)

    def map_corpus(self, documents, directory=None, n_job=-1):
        """
        Metamaps a large amount of files quickly by forking processes and utilizing multiple cores

        :param documents: an array of documents to map
        :param directory: location to map all files
        :param n_job: number of cores to utilize at once while mapping - this may use a large amount of memory
        :return:
        """


        raise NotImplementedError() #TODO implement utilizing code for parallel process mapper from n2c2


    def _run_metamap(self, args, document):
        """
        Runs metamap through bash and feeds in appropriate arguments
        :param args: arguments to feed into metamap
        :param document: the raw text to be metamapped
        :return:
        """

        bashCommand = 'bash %s %s' % (self.metamap_path, args)
        process = subprocess.Popen(bashCommand, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate(input=bytes(document, 'UTF-8'))
        output = str(output.decode('utf-8'))

        xml = ""
        for line in output.split("\n")[1:]:
            if 'DOCTYPE' not in line and 'xml' not in line:
                xml += line+'\n'
        xml = "<metamap>\n" + xml + "</metamap>"  # surround in single root tag - hacky.
        xml = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE MMOs PUBLIC "-//NLM//DTD MetaMap Machine Output//EN" "http://metamap.nlm.nih.gov/DTD/MMOtoXML_v5.dtd">\n'+xml

        if output is None:
            raise Exception("An error occured while using metamap: %s" % error)


        dict = xmltodict.parse(xml)
        return dict

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


        utterances = metamap_dict['metamap']['MMOs']['MMO']['Utterances']['Utterance']
        mapped_terms = []



        mapped_terms = list(self._item_generator(metamap_dict, 'Candidate'))

        all_terms = []

        for term in mapped_terms:
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
        :param label: the label to assign to each annotation, defaults to first semantic type of mapped_term
        :return: a annotation formatted to spacy's specifications
        """

        annotations = {'entities': {}}

        count = 0
        for term in mapped_terms:
            for span in self.get_span_by_term(term): #if a single entity corresonds to a disjunct span

                entity_start, entity_end = span
                if entity_label is None:
                    annotations['entities'][count] = (entity_start, entity_end, self.get_semantic_types_by_term(term)[0])
                else:
                    annotations['entities'][count] = (entity_start, entity_end, entity_label)
                count+=1

        return annotations





    def get_term_by_semantic_type(self, mapped_terms, include=[], exclude=None):
        """
        Returns Metamapped utterances that all contain a given set of semantic types found in include

        :param mapped_terms: An array of candidate dictionaries
        :return: the dictionaries that contain a term with all the semantic types in semantic_types
        """

        if exclude is not None:
            intersection = set(include).intersection(exclude)
            if intersection:
                raise Exception("Include and exclude overlap with the following semantic types: "+", ".join(intersection))
        matches = []

        for term in mapped_terms:

            found_types = []

            if int(term['SemTypes']['@Count']) == 0:
                continue

            if int(term['SemTypes']['@Count']) == 1:
                found_types.append(term['SemTypes']['SemType'])

            if int(term['SemTypes']['@Count']) > 1:
                found_types = term['SemTypes']['SemType']


            if exclude is not None and set(exclude).issubset(set(found_types)):
                continue

            if set(include).issubset(set(found_types)):
                matches.append(term)


        return matches

    def get_span_by_term(self,term):
        """
        Takes a given utterance dictionary (term) and extracts out the character indices of the utterance

        :param term: The full dictionary corresponding to a metamap term
        :return: the span of the referenced term in the document
        """
        if int(term['ConceptPIs']['@Count']) == 1:
            start = int(term['ConceptPIs']['ConceptPI']['StartPos'])
            length = int(term['ConceptPIs']['ConceptPI']['Length'])
            return [(start, start+length)]

        spans = []
        for span in term['ConceptPIs']['ConceptPI']:
            start = int(span['StartPos'])
            length = int(span['Length'])
            spans.append((start, start+length))
        return spans

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






