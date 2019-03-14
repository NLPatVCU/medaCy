"""
Converts data from brat to con. Enter input and output directories as command line arguments.
Each '.ann' file must have a '.txt' file in the same directory with the same name, minus the extension.
Use '-c' (without quotes) as an optional final command-line argument to copy the text files used
in the conversion process to the output directory.

Also possible to import 'convert_brat_to_con()' directly and pass the paths to the ann and txt files
for individual conversion.

:author: Steele W. Farnsworth
:date: 13 March, 2019
"""

from sys import argv
from re import split, fullmatch, DOTALL, findall
from medacy.tools.converters.conversion_tools.line import Line
import re
import os
import shutil
import logging
import tabulate


# A regex pattern for consecutive whitespace other than a new line character
whitespace_pattern = re.compile("( +|\t+)+")
# Regex pattern for BRAT T annotations
brat_pattern_T = r"T\d+\t\S+ \d+ \d+\t.+"

# Used for stats at the end
num_lines = 0
num_skipped_regex = 0


def is_valid_brat(item: str):
    """Returns a boolean value for whether or not a given line is in the BRAT format."""
    # Define the regex pattern for BRAT.
    # Note that this pattern allows for three to six spaces to count as a tab
    if not isinstance(item, str): return False
    if fullmatch(brat_pattern_T, item, DOTALL): return True
    else: return False


def line_to_dict(item):
    """
    Converts a string that is a line in brat format to a dictionary representation of that data.
    Keys are: id_type, id_num, data_type, start_ind, end_ind, data_type.
    :param item: The line of con text (str).
    :return: The dictionary containing that data.
    """
    split1 = split("\t", item)
    split2 = split(" ", split1[1])
    split3 = [split1[0]] + split2 + [split1[2]]
    s = [i.strip() for i in split3]  # remove whitespace
    return {"id_type": s[0][0], "id_num": int(s[0][1:]), "data_type": s[1], "start_ind": int(s[2]),
            "end_ind": int(s[3]), "data_item": s[4]}


def switch_extension(name, ext):
    """
    Primarily for internal use.
    Takes the name of a file (str) and changes the extension to the one provided (str)
    """
    return os.path.splitext(name)[0] + ext


def find_line_num(text_, start):
    """
    :param text_: The text of the file, ex. f.read()
    :param start: The index at which the desired text starts
    :return: The line index (starting at 0) containing the given start index
    """
    return text_[:int(start)].count("\n")


def get_word_num(line_obj: Line, entity_index):
    """
    Returns the word number relative to the start of the line, with counting starting at 0,
    of the first word of the entity.
    :param line_obj: The Line that the entity occurs in.
    :param entity_index: The absolute index of the entity, given by the annotation.
    :return: The word index of the entity.
    """
    index_within_line = entity_index - line_obj.index
    substring_before_entity = line_obj.text[:index_within_line]
    matched_spaces = findall(whitespace_pattern, substring_before_entity)
    return len(matched_spaces)


def convert_brat_to_con(brat_file_path, text_file_path=None):
    """
    Takes a path to a brat file and returns a string representation of that file converted to the con format.
    :param brat_file_path: The path to the brat file; not the file itself. If the path is not valid, the argument
        will be assumed to be text of the brat file itself.
    :param text_file_path: The path to the text file; if not provided, assumed to be a file with the same path as
        the brat file ending in '.txt' instead of '.ann'. If neither file is found, raises error.
    :return: A string (not a file) of the con equivalent of the brat file.
    """

    global num_lines, num_skipped_regex

    # By default, find txt file with equivalent name
    if text_file_path is None:
        text_file_path = switch_extension(brat_file_path, ".txt")
        if not os.path.isfile(text_file_path):
            raise FileNotFoundError("No text file path was provided and no matching text file was found in the input"
                                    " directory")
        with open(text_file_path, 'r') as text_file:
            text = text_file.read()
            text_lines = Line.init_lines(text)
    # Otherwise open the file with the path passed to the function
    elif os.path.isfile(text_file_path):
        with open(text_file_path, 'r') as text_file:
            text = text_file.read()
            text_lines = Line.init_lines(text)
    else: raise FileNotFoundError("No text file path was provided or the file was not found."
                                  " Note that direct string input of the source text is not supported.")

    # If con_file_path is actually a path, open it and split it into lines
    if os.path.isfile(brat_file_path):
        with open(brat_file_path, 'r') as brat_file:
            brat_text = brat_file.read()
            brat_text_lines = brat_text.split('\n')
    else:  # Else, read whatever string is passed to the function as if it were the file itself
        brat_text = brat_file_path
        brat_text_lines = brat_text.split('\n')

    output_lines = ""  # This value will be appended

    for line in brat_text_lines:

        if line.startswith("#") or not line:
            # Comments and blank lines can be skipped without warning
            continue
        elif not is_valid_brat(line):
            logging.warning("Incorrectly formatted line in %s was skipped: \"%s\"." % (brat_file_path, line))
            num_skipped_regex += 1
            continue

        d = line_to_dict(line)

        start_line_num = find_line_num(text, d["start_ind"])
        start_source_line = text_lines[start_line_num]
        start_word_num = get_word_num(start_source_line, d["start_ind"])
        start_str = str(start_line_num + 1) + ':' + str(start_word_num)

        end_line_num = find_line_num(text, d["end_ind"])
        end_word_num = start_word_num + len(re.findall(whitespace_pattern, d["data_item"]))
        end_str = str(end_line_num + 1) + ':' + str(end_word_num)

        con_line = "c=\"%s\" %s %s||t=\"%s\"\n" % (d["data_item"], start_str, end_str, d['data_type'])
        output_lines += con_line

        num_lines += 1

    return output_lines


if __name__ == '__main__':

    # Get the input and output directories from the command line.

    if len(argv) < 3:
        # Command-line arguments must be provided for the input and output directories.
        raise IOError("Please run the program again, entering the input and output directories as command-line"
                      " arguments in that order. Optionally, enter '-c' as a final command line argument if you want"
                      " to copy the text files used in the conversion over to the output directory.")

    try:
        input_dir_name = argv[1]
        input_dir = os.listdir(input_dir_name)
    except FileNotFoundError:  # dir doesn't exist
        while not os.path.isdir(input_dir_name):
            input_dir_name = input("Input directory not found; please try another directory:")
        input_dir = os.listdir(input_dir_name)
    try:
        output_dir_name = argv[2]
        output_dir = os.listdir(output_dir_name)
    except FileNotFoundError:
        while not os.path.isdir(output_dir_name):
            output_dir_name = input("Output directory not found; please try another directory:")
            output_dir = os.listdir(output_dir_name)

    # Create a list of only the .txt files in the input directory
    text_files = [f for f in input_dir if f.endswith(".txt")]
    # Create a list of all .ann files in the input directory that have a txt equivalent
    ann_files = [f for f in input_dir if f.endswith(".ann") and switch_extension(f, ".txt") in text_files]

    # Ensure user is aware if there are no files to convert
    if len(ann_files) < 1:
        raise FileNotFoundError("There were no ann files in the input directory with a corresponding text file. "
                                "Please ensure that the input directory contains ann files and that each file has "
                                "a corresponding txt file (see help for this program).")

    # Create the log
    log_path = os.path.join(output_dir_name, "conversion.log")
    logging.basicConfig(filename=log_path, level=logging.WARNING)

    for input_file_name in ann_files:
        full_file_path = os.path.join(input_dir_name, input_file_name)
        output_file_name = switch_extension(input_file_name, ".con")
        content = convert_brat_to_con(full_file_path)
        with open(os.path.join(output_dir_name, output_file_name), "a+") as output_file:
            output_file.write(content)

    # Paste all the text files used in the conversion process to the output directory
    # if there's a fourth command line argument and that argument is -c
    if len(argv) >= 4 and argv[3] == "-c":
        text_files_with_match = [f for f in text_files if switch_extension(f, ".ann") in ann_files]
        for f in text_files_with_match:
            full_name = os.path.join(input_dir_name, f)
            shutil.copy(full_name, output_dir_name)

    # Compile and print stats to log
    stat_headers = ["Total lines", "Total converted",
                    "Skipped did not match regex", "Percent converted"]

    stat_data = [
        num_lines,
        num_lines - num_skipped_regex,
        num_skipped_regex,
        (num_lines - num_skipped_regex) / num_lines
    ]

    conversion_stats = tabulate.tabulate(headers=stat_headers, tabular_data=[stat_data])
    logging.warning("\n" + conversion_stats)
