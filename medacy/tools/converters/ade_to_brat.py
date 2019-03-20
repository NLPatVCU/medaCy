"""
Converts data from ade to brat. Enter input and output directories as command line arguments.
The associated txt file is not needed to perform any calculations.

:date: 23 November, 2018
:author: Steele W. Farnsworth
"""

from sys import argv as cmd_arg
from re import split
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


def get_data_types(title):
    """Each input file name specifies two data types separated by a hyphen;
    this function returns a tuple of those two strings."""
    name = os.path.splitext(title)[0]  # get the file name w/o the extension
    types = split("-", name)
    return types


# get only rel files in the input dir
rel_files = [f for f in input_dir if f.endswith(".rel")]

for file_name in rel_files:
    data_types = get_data_types(file_name)
    with open(input_dir_name + '/' + file_name) as input_file:
        output_file_name = os.path.splitext(file_name)[0] + ".ann"  # creates a new file name, changing .rel to .ann
        output_file = open(output_dir_name + '/' + output_file_name, "a+")
        for line in input_file:
            split_line = [item.rstrip() for item in split("\|", line)]  # remove whitespace characters
            item_id = split_line[0]
            data1_entry = split_line[2]; span1_start = split_line[3]; span1_end = split_line[4]
            data2_entry = split_line[5]; span2_start = split_line[6]; span2_end = split_line[7]
            line1 = "T" + item_id + " " + data_types[1] + " " + span1_start + " " + span1_end + "\t" + data1_entry + "\n"
            line2 = "T" + item_id + " " + data_types[0] + " " + span2_start + " " + span2_end + "\t" + data2_entry + "\n"
            output_file.write(line1)
            output_file.write(line2)
        output_file.close()

