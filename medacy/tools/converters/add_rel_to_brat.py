"""
:author: Steele W. Farnsworth
:date: 28 May, 2019
"""

import os
import re
from sys import argv
from medacy.tools.converters.conversion_tools.line import Line
from medacy.tools.converters.con_to_brat import get_absolute_index, switch_extension
from medacy.tools.converters.brat_to_con import line_to_dict

class Entity:
    """Represents an entity from an ann file."""
    def __init__(self, t: int, ent_type: str, start: int, end: int, text: str):
        self.ent_type = ent_type
        self.start = start
        self.end = end
        self.text = text
        self.t = t

    def __eq__(self, other):
        return True if self.start == other.start and self.end == other.end and self.text == other.text else False

    def __str__(self):
        return "T%i\t%s %i %i\t%s\n" % (self.t, self.ent_type, self.start, self.end, self.text)


def add_rel_to_brat(ann_path, rel_path, txt_path, null_ent_1="null", null_ent_2="null"):
    """
    Given the full paths for an ann file, a rel file, and the txt file that those annotations represent,
    converts the incoming rel data to brat and appends it to the ann file. Input ann file must not contain any
    rel data.

    rel annotations typically refer to entities that have already been annotated. If this is not the case,
    new entities are appended to the ann file. However, the rel format only allows us to determine the character
    spans and entity text and not the entity type. Two parameters exist for this function to specify what those
    entities types should be.

    :param ann_path: The full path to the ann file
    :param rel_path: The full path to the rel file
    :param txt_path: The full path to the text file
    :param null_ent_1: What type of entity the first item in a relation pair should be called if not found in the
        input file.
    :param null_ent_2: What type of entity the second item in a relation pair should be called if not found in the
        input file.
    :return:
    """

    # Get the text of the old ann file
    with open(ann_path, "r") as f:
        old_ann_text = f.read()

    old_ann_text_lines = old_ann_text.split("\n")

    t = 0

    # Go to the last line of the old ann file to figure out what T number we're starting at
    # Ask me if this doesn't make sense
    for line in reversed(old_ann_text_lines):
        if line.startswith("T"):
            t = int(re.findall("T\d+", line)[0][1:]) + 1
            break

    # We need to have a list of all the entities so that we can find the match
    all_entities = []

    # For each line in the old ann file that starts with T, we need to make a new Entity object and
    # add it to the list of entities
    for line in old_ann_text_lines:
        if line.startswith("T"):
            d = line_to_dict(line)
            new_entity = Entity(d["id_num"], d["data_type"], d["start_ind"], d["end_ind"], d["data_item"])
            all_entities.append(new_entity)

    # Get the text of the rel file
    with open(rel_path, "r") as f:
        rel_text = f.read()

    # Get the text file that we have the annotations of
    with open(txt_path, "r") as f:
        text = f.read()
    # Create line objects for that file; don't worry about how this works
    text_lines = Line.init_lines(text)

    # This is the string we'll be appending
    output_text = ""
    r = 1

    # Iterate over all the relation lines
    for line in rel_text.split("\n"):
        if line == "": continue  # Skip blank lines
        # Using regex to pick apart the line of input
        c_items = re.findall(r'c="([^"]*)"', line)
        c1, c2 = c_items[0], c_items[1]
        all_spans = re.findall(r'\d+:\d+', line)
        r_item = re.findall(r'r="([^"]*)"', line)[0]

        start_ind_1 = get_absolute_index(text_lines, all_spans[0], c1)
        start_ind_2 = get_absolute_index(text_lines, all_spans[2], c2)

        # Create new Entity objects for the incoming data, which don't come with T numbers or entity types
        new_ent_1 = Entity(0, null_ent_1, start_ind_1, start_ind_1 + len(c1), c1)
        new_ent_2 = Entity(0, null_ent_2, start_ind_2, start_ind_2 + len(c2), c2)

        # These are booleans for if we're dealing with new data. We're going to check to see if these entities
        # already appear in our data.
        new_ent_1_new = True
        new_ent_2_new = True

        # Loop through to see if our new entities are already in the data.
        for e in all_entities:
            if new_ent_1 == e:
                new_ent_1 = e
                new_ent_1_new = False
                break
        for e in all_entities:
            if new_ent_2 == e:
                new_ent_2 = e
                new_ent_2_new = False
                break

        # Set the T values if they are new data and add it to the output text
        if new_ent_1_new:
            new_ent_1.t = t
            t += 1
            output_text += str(new_ent_1)
        if new_ent_2_new:
            new_ent_2.t = t
            t += 1
            output_text += str(new_ent_1)

        # Whether the entities are new or not, we still need to add the new relationship data
        output_text += "R%i\t%s Arg1:T%i Arg2:T%i\n" % (r, r_item, new_ent_1.t, new_ent_2.t)
        r += 1

    # Write the new data to file, and we're done
    with open(ann_path, "a") as f:
        f.write(output_text)

def main(cmd_arg: list):

    if len(cmd_arg) < 3:
        raise IOError(""""Please run the program again with command line arguments for two directories:
                            1) Containing the ann and txt files
                            2) Containing the new rel files""")

    ann_txt_dir_path, new_rel_dir_path = cmd_arg[1], cmd_arg[2]

    ann_file_names = [f for f in os.listdir(ann_txt_dir_path) if f.endswith(".ann")]

    all_file_tuples = []

    for ann in ann_file_names:
        full_ann_path = os.path.join(ann_txt_dir_path, ann)
        full_txt_path = os.path.join(ann_txt_dir_path, switch_extension(ann, ".txt"))
        full_rel_path = os.path.join(new_rel_dir_path, switch_extension(ann, ".rel"))

        if not os.path.isfile(full_txt_path):
            raise FileNotFoundError("Ann file %s did not have an associated txt file." % ann)
        if not os.path.isfile(full_rel_path):
            raise FileNotFoundError("Ann file %s did not have an associated rel file." % ann)

        new_tuple = (full_ann_path, full_rel_path, full_ann_path)
        all_file_tuples.append(new_tuple)

    for a, r, t in all_file_tuples:
        add_rel_to_brat(a, r, t)


if __name__ == "__main__":
    main(argv)