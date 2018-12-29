"""
Converts data from con to brat. Enter input and output directories as command line arguments.
Each '.con' file must have a '.txt' file in the same directory with the same name, minus the extension.

:author: Steele W. Farnsworth
:date: 2 December, 2018
"""

from sys import argv as cmd_arg
from re import split, findall
import os


# Get the input and output directories from the command line.

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
    """Converts a line of con data to a dict"""
    c_item = findall(r'c="([^"]*)"', item)
    t_item = findall(r't="([^"]*)"', item)
    spans = findall(r'\d+:\d+', item)
    items = {"data_item": c_item[0], "start_ind":  spans[0], "end_ind": spans[1], "data_type": t_item[0]}
    return items


def switch_extension(name, ext):
    """Takes the name of a file (str) and changes the extension to the one provided (str)"""
    return os.path.splitext(name)[0] + ext


def get_absolute_index(txt, txt_lns, ind):
    """
    Given one of the \d+:\d+ spans, which represent the index of a char relative to the start of the line it's on,
    returns the index of that char relative to the start of the file.
    :param txt: The text file associated with the annotation.
    :param txt_lns: The same text file as a list broken by lines
    :param ind: The string in format \d+:\d+
    :return: The absolute index
    """

    # convert ind to line_num and char_num
    nums = split(":", ind)
    line_num = int(nums[0]) - 1  # line nums in con start at 1 and not 0
    char_num = int(nums[1])

    this_line = txt_lns[line_num]
    line_index = txt.index(this_line)  # get the absolute index of the entire line
    abs_index = line_index + char_num
    return abs_index


# get only the con files in input_dir
con_files = [f for f in input_dir if f.endswith(".con")]

for input_file_name in con_files:
    """Get the txt file associated with the ann file, if it exists"""
    text_file_name = switch_extension(input_file_name, ".txt")
    try:
        text_file = open(input_dir_name + '/' + text_file_name, "r")
    except FileNotFoundError:
        print("'{}' does not have an associated '.txt' file; it has been skipped.".format(input_file_name))
        continue  # skip this iteration of the for loop if the file doesn't exist

    text = text_file.read()
    text_file.close()
    text_lines = split("\n", text)

    """Create the output file"""
    output_file_name = switch_extension(input_file_name, ".ann")
    output_file = open(output_dir_name + '/' + output_file_name, "a+")

    with open(input_dir_name + '/' + input_file_name) as input_file:
        t = 1
        for line in input_file:
            d = line_to_dict(line)
            start_ind = get_absolute_index(text, text_lines, d["start_ind"])
            span_length = d["data_item"].__len__()
            end_ind = start_ind + span_length
            output = "T" + str(t) + "\t" + d["data_type"] + " " + str(start_ind) + " " + str(end_ind) + "\t" \
                     + d["data_item"] + "\n"
            output_file.write(output)
            t += 1

    output_file.close()
