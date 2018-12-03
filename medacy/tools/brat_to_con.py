"""
Converts data from brat to con. Enter input and output directories as command line arguments.
Each '.ann' file must have a '.txt' file in the same directory with the same name, minus the extension.

:author: Steele W. Farnsworth
:date: 27 November, 2018
"""

from sys import argv as cmd_arg
from re import split
import os


"""Get the input and output directories from the command line."""

try:
    input_dir_name = cmd_arg[1]
    input_dir = os.listdir(input_dir_name)
except FileNotFoundError:  # dir doesn't exist
    while not os.path.isdir(input_dir_name):
        input_dir_name = input("Input directory not found; please try another directory:")
    input_dir = os.listdir(input_dir_name)
except IndexError:  # user didn't enter args at cmd
    input_dir_name = "null"  # ensure next line is false
    while not os.path.isdir(input_dir_name):
        input_dir_name = input("Input directory not found; please try another directory:")
    input_dir = os.listdir(input_dir_name)
try:
    output_dir_name = cmd_arg[2]
    output_dir = os.listdir(output_dir_name)
except FileNotFoundError:
    while not os.path.isdir(output_dir_name):
        output_dir_name = input("Output directory not found; please try another directory:")
    output_dir = os.listdir(output_dir_name)
except IndexError:
    output_dir_name = "null"
    while not os.path.isdir(output_dir_name):
        output_dir_name = input("Output directory not found; please try another directory:")
    output_dir = os.listdir(output_dir_name)


def line_to_dict(item):
    """Converts a given line from a file into a dict of each item in that line."""
    split1 = split("\t", item)
    split2 = split(" ", split1[1])
    split3 = [split1[0], *split2, split1[2]]
    s = [i.rstrip() for i in split3]  # remove whitespace
    return {"T": s[0], "data_type": s[1], "start_ind": int(s[2]), "end_ind": int(s[3]), "data_item": s[4]}


def switch_extension(name, ext):
    """Takes the name of a file (str) and changes the extension to the one provided (str)"""
    return os.path.splitext(name)[0] + ext


def switch_format(items, start, end):
    """
    :param items: The dict of the original brat input data
    :param start: The start indices for the data item, which are calculated by this program
    :param end: The end indices, also
    :return: The data formatted according to the con data format
    """
    return "c=\"" + items["data_item"] + '\" ' + start + " " + end + '||t=\"' + items["data_type"] + '\"\n'


def find_line_num(text_, start):
    """
    :param text_: The text of the file, ex. f.read()
    :param start: The index at which the desired text starts
    :return: The line index (starting at 0) containing the given start index
    """
    return text_[:int(start)].count("\n")


def get_relative_index(text_: str, line_, absolute_index):
    """
    Takes the index of a phrase (the phrase itself is not a parameter) relative to the start of its
    file and returns its index relative to the start of the line that it's on.
    :param text_: The text of the file, not separated by lines
    :param line_: The text of the line being searched for
    :param absolute_index: The index of a given phrase
    :return: The index of the phrase relative to the start of the line
    """
    line_index = text_.index(line_)
    return int(absolute_index) - line_index


def get_end_word_index(data_item: str, start_index, end_index):
    """Returns the index of the first char of the last word of data_item_;
    all parameters shadow the appropriate name in the final for loop"""
    words = split(" ", data_item)
    if words.__len__() == 1:
        return start_index  # If there's only one word, the start of the first word is the start of the last word
    else:
        last_word = words[-1]
        return end_index - last_word.__len__()


# Creates a list of only the .ann files in the input directory
ann_files = [f for f in input_dir if f.endswith(".ann")]

for input_file_name in ann_files:

    """Get the txt file associated with the ann file, if it exists"""
    txt_file_name = switch_extension(input_file_name, ".txt")
    try:
        txt_file = open(input_dir_name + '/' + txt_file_name, "r")
    except FileNotFoundError:
        print("'{}' does not have an associated '.txt' file; it has been skipped.".format(input_file_name))
        continue  # skip this iteration of the for loop if the file doesn't exist

    text = txt_file.read()
    txt_file.close()
    txt_lines = split("\n", text)

    """Create the output file"""
    output_file_name = switch_extension(input_file_name, ".con")
    output_file = open(output_dir_name + '/' + output_file_name, "a+")

    with open(input_dir_name + '/' + input_file_name) as input_file:
        for line in input_file:
            d = line_to_dict(line)

            start_line_num = find_line_num(text, d["start_ind"])
            start_char_num = get_relative_index(text, txt_lines[start_line_num], d["start_ind"])
            start_str = str(start_line_num + 1) + ':' + str(start_char_num)

            """Note that the end word has an extra calculation because the index of the first char
            of the last word is what is needed, not the last char of the last word."""
            end_line_num = find_line_num(text, d["end_ind"])
            end_char_num = get_relative_index(text, txt_lines[end_line_num], d["end_ind"])
            end_word_num = get_end_word_index(d["data_item"], start_char_num, end_char_num)
            end_str = str(end_line_num + 1) + ':' + str(end_word_num)

            output_line = switch_format(d, start_str, end_str)
            output_file.write(output_line)

    output_file.close()
