"""
Contains a function for adding assertion annotations from i2b2 data
to pre-existing BRAT annotations.

Assertion annotations look like this:
c=”prostate cancer” 5:7 5:8||t=”problem”||a=”present”

The desired output looks like this:
T1     problem  55 70     prostate cancer
A1     present T1
"""

import argparse
import logging
import os
import re

from medacy.tools.converters.brat_to_con import switch_extension
from medacy.tools.converters.con_to_brat import get_absolute_index, line_to_dict
from medacy.tools.converters.conversion_tools.line import Line
from medacy.tools.entity import Entity

assertion_pattern = r'c="([^"]*)" \d+:\d+ \d+:\d+\|\|t="([^"]*)"\|\|a="([^"]*)"'


def is_valid_assert(sample):
    """Validates that a line is a valid assertion annotation"""
    return isinstance(sample, str) and re.fullmatch(assertion_pattern, sample)


def add_ast_to_brat(ast_file_path, ann_file_path, txt_file_path):
    """
    Adds the assertion annotations to a given ann file
    :param ast_file_path: The assertion file to get the assertion annotations from
    :param ann_file_path: The ann file to add the assertion annotations to
    :param txt_file_path: The text file that the previous two are annotating
    :return: None
    """

    with open(txt_file_path) as f:
        text = f.read()
    text_lines = Line.init_lines(text)

    with open(ast_file_path) as f:
        ast_text = f.read()

    if ast_text == "":
        logging.info(f"There were no assertions in file {ast_file_path}, no conversion was performed")
        return

    assertions = ast_text.split('\n')
    entities = Entity.init_from_doc(ann_file_path)

    a = 1  # used to keep track of the assertion number
    add_to_ann = ""

    for line in assertions:

        if not is_valid_assert(line):
            logging.warning(f"Invalid line of ast text in file {ast_file_path} was skipped: {line}")
            continue

        # Get the part of the assertion annotation that is an entity (up to the '||a')
        print(line)
        a_part_index = line.index('||a')
        assertion_text = line[a_part_index + 5:-1]
        entity_part = line[:a_part_index]
        # Break up entity_part into named substrings
        ent_dict = line_to_dict(entity_part)
        # Get the BRAT-formatted (relative to the start of the document) index for the start of the entity
        ent_text = ent_dict['data_item']
        ent_type = ent_dict['data_type']
        start_ind = get_absolute_index(text_lines, ent_dict['start_ind'], ent_text)
        end_ind = start_ind + len(ent_text)

        # Get the text of the entity as it appears in the document, since it might not match what's provided
        # in the assertion file
        real_ent_text = text[start_ind:end_ind]
        if real_ent_text != ent_text:
            logging.info(f"Enity text in document {ast_file_path} didn't match; expected '{ent_text}', actual {real_ent_text}")
        ent_text = real_ent_text

        ent = Entity(ent_type, start_ind, end_ind, ent_text)
        ent_match = None

        # See if the entity already exists in the ann file
        for e in entities:
            if ent == e:
                ent_match = e
                break

        # If not, add it to the new entities
        if ent_match is None:
            add_to_ann += str(ent) + '\n'
        else:
            # If the ent does have a match, we will use that from now on instead of the one we made
            ent = ent_match

        add_to_ann += f"A{a}\t{assertion_text} T{ent.num}\n"
        a += 1
        # End for

    with open(ann_file_path, 'a') as f:
        print("WRITING", add_to_ann)
        f.write(add_to_ann)


def main():
    description = """Add assertion annotations to BRAT ann files. Files that refer to the same document must all have
    the same name, except for the file extensions."""
    parser = argparse.ArgumentParser(prog='Add Assert to BRAT', description=description)
    parser.add_argument('-a', '--ast_file_dir', required=True, help='Directory containing the assertion files.')
    parser.add_argument('-t', '--txt_file_dir', required=True, help='Directory containing the text files.')
    parser.add_argument('-b', '--ann_file_dir', required=True, help='Directory containing the BRAT ann files.')
    args = parser.parse_args()

    file_tuples = []

    ast_files = os.listdir(args.ast_file_dir)

    for file_name in ast_files:
        txt_file_name = switch_extension(file_name, ".txt")
        txt_file_path = os.path.join(args.txt_file_dir, txt_file_name)

        ann_file_name = switch_extension(file_name, ".ann")
        ann_file_path = os.path.join(args.ann_file_dir, ann_file_name)

        ast_file_path = os.path.join(args.ast_file_dir, file_name)

        file_tuples.append((ast_file_path, ann_file_path, txt_file_path))

    for a, b, t in file_tuples:
        add_ast_to_brat(a, b, t)


if __name__ == '__main__':
    main()
