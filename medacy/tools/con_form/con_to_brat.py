"""
Converts data from con to brat. Enter input and output directories as command line arguments.
Each '.con' file must have a '.txt' file in the same directory with the same name, minus the extension.
Use '-c' (without quotes) as an optional final command-line argument to copy the text files used
in the conversion process to the output directory.

Function 'convert_con_to_brat()' can be imported independently and run on individual files.

This version does not produce accurate output. Revisions are underway.

:author: Steele W. Farnsworth
:date: 18 February, 2019
"""

from sys import argv as cmd_arg, exit
from re import split, findall, fullmatch, DOTALL
import os
import shutil


def is_valid_con(item: str):
    """
    Comprehensively tests to see if a given line is in valid con format. Returns respective boolean value.
    :param item: A string that is a line of text, hopefully in the con format.
    :return: Boolean of whether or not the line matches a con regular expression.
    """
    if not isinstance(item, str): return False
    con_pattern = "c=\".+?\" \d+:\d+ \d+:\d+\|\|t=\".+?\"(|\n)"
    if fullmatch(con_pattern, item): return True
    else: return False


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

    # By default, find txt file with equivalent name
    if text_file_path is None:
        text_file_path = switch_extension(con_file_path, ".txt")
        if not os.path.isfile(text_file_path):
            raise FileNotFoundError("No text file path was provided and no matching text file was found in the input"
                                    " directory")
        with open(text_file_path, 'r') as text_file:
            text = text_file.read()
            text_lines = text.split('\n')
    # Else, open the file with the path passed to the function
    elif os.path.isfile(text_file_path):
        with open(text_file_path, 'r') as text_file:
            text = text_file.read()
            text_lines = text.split('\n')
    else: raise FileNotFoundError("No text file path was provided or the file was not found."
                                  " Note that direct string input of the source text is not supported.")

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
        if not is_valid_con(line): continue
        d = line_to_dict(line)
        start_ind = get_absolute_index(text, text_lines, d["start_ind"])
        span_length = d["data_item"].__len__()
        end_ind = start_ind + span_length
        output_line = "T%s\t%s %s %s\t%s\n" % (str(t), d["data_type"], str(start_ind), str(end_ind), d["data_item"])
        output_text += output_line
        t += 1

    return output_text


if __name__ == '__main__':

    # Get the input and output directories from the command line.

    if not cmd_arg.__len__() >= 3:
        # Command-line arguments must be provided for the input and output directories.
        # Else, prints instructions and aborts the program.
        print("Please run the program again, entering the input and output directories as command-line arguments"
              " in that order. Optionally, enter '-c' as a final command line argument if you want to copy"
              " the text files used in the conversion over to the output directory.")
        exit()

    try:
        input_dir_name = cmd_arg[1]
        input_dir = os.listdir(input_dir_name)
    except FileNotFoundError:  # dir doesn't exist
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

    # Get only the text files in input_dir
    text_files = [f for f in input_dir if f.endswith(".txt")]
    # Get only the con files in input_dir that have a ".txt" equivalent
    con_files = [f for f in input_dir if f.endswith(".con") and switch_extension(f, ".txt") in text_files]

    for input_file_name in con_files:
        full_file_path = os.path.join(input_dir_name, input_file_name)
        output_file_name = switch_extension(input_file_name, ".ann")
        output_file_path = os.path.join(output_dir_name, output_file_name)
        content = convert_con_to_brat(full_file_path)
        with open(output_file_path, "a+") as output_file:
            output_file.write(content)

    # Paste all the text files used in the conversion process to the output directory
    # if there's a fourth command line argument and that argument is -c
    if cmd_arg.__len__() == 4 and cmd_arg[3] == "-c":
        text_files_with_match = [f for f in text_files if switch_extension(f, ".con") in con_files]
        for f in text_files_with_match:
            full_name = os.path.join(input_dir_name, f)
            shutil.copy(full_name, output_dir_name)
