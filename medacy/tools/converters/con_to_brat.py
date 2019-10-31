"""
Converts data from con to brat. Enter input and output directories as command line arguments.
Each '.con' file must have a '.txt' file in the same directory with the same name, minus the extension.
Use '-c' (without quotes) as an optional final command-line argument to copy the text files used
in the conversion process to the output directory.

Function 'convert_con_to_brat()' can be imported independently and run on individual files.

This program can be used for conversion independently from medaCy if the Line class is copied
and pasted into a copy of this program.

:author: Steele W. Farnsworth
"""

import logging
import os
import re
import shutil
from re import split, findall, fullmatch
from sys import argv

import tabulate

from medacy.tools.converters.conversion_tools.line import Line

# Regex patterns
whitespace_pattern = "( +|\t+)+"
con_pattern = r"c=\".+?\" \d+:\d+ \d+:\d+\|\|t=\".+?\"(|\n)"

# Used for stats at the end
num_lines = 0
num_skipped_regex = 0
num_skipped_value_error = 0


def is_valid_con(item: str):
    """
    Comprehensively tests to see if a given line is in valid con format. Returns respective boolean value.
    :param item: A string that is a line of text, hopefully in the con format.
    :return: Boolean of whether or not the line matches a con regular expression.
    """
    return isinstance(item, str) and fullmatch(con_pattern, item)


def line_to_dict(item):
    """
    Converts a string that is a line in con format to a dictionary representation of that data.
    Keys are: data_item; start_ind; end_ind; data_type.
    :param item: The line of con text (str).
    :return: The dictionary containing that data.
    """
    c_item = findall(r'c="([^"]*)"', item)
    t_item = findall(r't="([^"]*)"', item)
    spans = findall(r'\d+:\d+', item)
    items = {"data_item": c_item[0], "start_ind": spans[0], "end_ind": spans[1], "data_type": t_item[0]}
    return items


def switch_extension(name, ext):
    """Takes the name of a file (str) and changes the extension to the one provided (str)"""
    return os.path.splitext(name)[0] + ext


def check_same_text(ent_text, start_ind, end_ind, doc_text):
    """
    Checks that the entity text in the BRAT annotations matches the text of the document at
    the indices calculated in this program; for example, if the casing is different in the
    txt document than in the ann file, this function would return the casing used in the txt
    document.

    :param ent_text: the text of the annotation in the CON annotation
    :param start_ind: the character index where the annotation starts
    :param end_ind: the character index where the annotaiton ends
    :param doc_text: the text of the txt document
    :return: the text between the indices in the document
    """

    # Get the text between the annotations in the document,
    # even if it's different from the provided text
    text_in_doc = doc_text[start_ind:end_ind]

    if ent_text == text_in_doc:
        return True

    return text_in_doc


def get_absolute_index(txt_lns, ind, entity):
    """
    Given one of the \d+:\d+ spans, which represent the index of a word relative to the start of the line it's on,
    returns the index of that char relative to the start of the file.
    :param txt_lns: The list of Line objects for that file.
    :param ind: The string in format \d+:\d+
    :param entity: The text of the entity
    :return: The absolute index
    """

    # Convert ind to line_num and char_num
    nums = split(":", ind)
    line_num = int(nums[0]) - 1  # line nums in con start at 1 and not 0
    word_num = int(nums[1])

    this_line = txt_lns[line_num]
    line_index = this_line.index

    # Get index of word following n space
    split_by_whitespace = split(whitespace_pattern, this_line.text)
    split_by_whitespace = [s for s in split_by_whitespace if s != '']
    split_by_ws_no_ws = [s for s in split_by_whitespace if not s.isspace()]
    all_whitespace = [s for s in split_by_whitespace if s.isspace()]

    # Adjust word_num if first character cluster is whitespace
    if split_by_whitespace[0].isspace():
        line_to_target_word = split_by_ws_no_ws[:word_num - 1]
    else:
        line_to_target_word = split_by_ws_no_ws[:word_num]

    num_non_whitespace = sum([len(w) for w in line_to_target_word])
    num_whitespace = sum([len(w) for w in all_whitespace[:word_num]])

    index_within_line = num_whitespace + num_non_whitespace
    line_to_start_index = this_line.text[index_within_line:]
    entity_pattern_escaped = re.escape(entity)
    entity_pattern_spaced = re.sub(r"\\\s+", "\\\s+", entity_pattern_escaped)

    try:
        # Search for entity regardless of case or composition of intermediate spaces
        # match = re.search(entity_pattern_spaced, this_line.text, re.IGNORECASE)[0]
        match = re.search(entity_pattern_spaced, line_to_start_index, re.IGNORECASE)[0]
        offset = line_to_start_index.index(match)  # adjusts if entity is not the first char in its "word"
    except (ValueError, TypeError):
        logging.warning("""Entity not found in its expected line:
        \t"%s"
        \t"%s"
        \tRevision of input data may be required; conversion for this item was skipped""" % (entity, this_line)
        )
        return -1

    return index_within_line + line_index + offset


def convert_con_to_brat(con_file_path, text_file_path=None):
    """
    Converts a con file to a string representation of a brat file.
    :param con_file_path: Path to the con file being converted. If a valid path is not provided but the argument is a
        string, it will be parsed as if it were a representation of the con file itself.
    :param text_file_path: Path to the text file associated with the con file. If not provided, the function will look
        for a text file in the same directory with the same name except for the extention switched to 'txt'.
        Else, raises error. Note that no conversion can be performed without the text file.
    :return: A string representation of the brat file, which can then be written to file if desired.
    """

    global num_lines, num_skipped_regex, num_skipped_value_error

    # By default, find txt file with equivalent name
    if text_file_path is None:
        text_file_path = switch_extension(con_file_path, ".txt")
        if not os.path.isfile(text_file_path):
            raise FileNotFoundError("No text file path was provided and no matching text file was found in the input"
                                    " directory")
        with open(text_file_path, 'r') as text_file:
            text = text_file.read()
            text_lines = Line.init_lines(text)
    # Else, open the file with the path passed to the function
    elif os.path.isfile(text_file_path):
        with open(text_file_path, 'r') as text_file:
            text = text_file.read()
            text_lines = Line.init_lines(text)
    else: raise FileNotFoundError("No text file path was provided or the file was not found."
                                  " Note that direct string input of the source text is not supported.")

    num_lines += len(text_lines)

    # If con_file_path is actually a path, open it and split it into lines
    if os.path.isfile(con_file_path):
        with open(con_file_path, 'r') as con_file:
            con_text = con_file.read()
            con_text_lines = con_text.split('\n')
    # Else, read whatever string is passed to the function as if it were the file itself
    else:
        con_text = con_file_path
        con_text_lines = con_text.split('\n')

    output_text = ""
    t = 1
    for line in con_text_lines:
        if line == "" or line.startswith("#"): continue
        elif not is_valid_con(line):
            logging.warning("Incorrectly formatted line in %s was skipped: \"%s\"." % (con_file_path, line))
            num_skipped_regex += 1
            continue
        d = line_to_dict(line)
        start_ind = get_absolute_index(text_lines, d["start_ind"], d["data_item"])
        if start_ind == -1:
            num_skipped_value_error += 1
            continue  # skips data that could not be converted
        span_length = len(d["data_item"])
        end_ind = start_ind + span_length

        # Check that the text of the annotation matches what's between its spans in the text document
        is_match = check_same_text(d['data_item'], start_ind, end_ind, text)
        if isinstance(is_match, str):
            logging.info(f"Annotation in file '{con_file_path}' did not match text between spans: '{d['data_item']}' != '{is_match}'")
            d['data_item'] = is_match

        output_line = "T%s\t%s %s %s\t%s\n" % (str(t), d["data_type"], str(start_ind), str(end_ind), d["data_item"])
        output_text += output_line
        t += 1

    return output_text


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

    # Create the log
    log_path = os.path.join(output_dir_name, "conversion.log")
    logging.basicConfig(filename=log_path, level=logging.INFO)

    # Get only the text files in input_dir
    text_files = [f for f in input_dir if f.endswith(".txt")]
    # Get only the con files in input_dir that have a ".txt" equivalent
    con_files = [f for f in input_dir if f.endswith(".con") and switch_extension(f, ".txt") in text_files]

    # Ensure user is aware if there are no files to convert
    if len(con_files) < 1:
        raise FileNotFoundError("There were no con files in the input directory with a corresponding text file. "
                                "Please ensure that the input directory contains ann files and that each file has "
                                "a corresponding txt file (see help for this program).")

    for input_file_name in con_files:
        full_file_path = os.path.join(input_dir_name, input_file_name)
        output_file_name = switch_extension(input_file_name, ".ann")
        output_file_path = os.path.join(output_dir_name, output_file_name)
        content = convert_con_to_brat(full_file_path)
        with open(output_file_path, "a+") as output_file:
            output_file.write(content)

    # Paste all the text files used in the conversion process to the output directory
    # if there's a fourth command line argument and that argument is -c
    if len(argv) >= 4 and argv[3] == "-c":
        text_files_with_match = [f for f in text_files if switch_extension(f, ".con") in con_files]
        for f in text_files_with_match:
            full_name = os.path.join(input_dir_name, f)
            shutil.copy(full_name, output_dir_name)

    # Compile and print stats to log
    stat_headers = ["Total lines", "Total converted", "Lines skipped", "Skipped due to value error",
                    "Skipped did not match regex", "Percent converted"]
    stat_data = [
        num_lines,
        num_lines - num_skipped_regex - num_skipped_value_error,
        num_skipped_regex + num_skipped_value_error,
        num_skipped_value_error,
        num_skipped_regex,
        (num_lines - num_skipped_regex - num_skipped_value_error) / num_lines
    ]

    conversion_stats = tabulate.tabulate(headers=stat_headers, tabular_data=[stat_data])
    logging.warning("\n" + conversion_stats)
