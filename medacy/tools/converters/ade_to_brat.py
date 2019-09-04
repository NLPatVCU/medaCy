"""
Converts data from ade to brat. Enter input and output directories as command line arguments.
The associated txt file is not needed to perform any calculations.

:date: 20 May, 2019
:author: Steele W. Farnsworth
"""

from sys import argv
import re
import os


class Entity:

    t = 1

    def __init__(self, ent_type: str, start: int, end: int, text: str):
        self.ent_type = ent_type
        self.start = start
        self.end = end
        self.text = text
        self.t = None

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end and self.text == other.text

    def add_t(self):
        self.t = self.__class__.t
        self.__class__.t += 1

    def __str__(self):
        assert self.t is not None, "This doesn't have a t number yet"
        return f"T{self.t}\t{self.ent_type} {self.start} {self.end}\t{self.text}\n"


made_pattern = "\d+\|.+\|.+\|\d+\|\d+\|.+\|\d+\|\d+"


def is_valid_ade(sample: str):
    """Returns True if line is valid made format; else return False"""
    if sample.startswith("#") or sample == "": return False
    if re.match(made_pattern, sample): return True


def line_to_dict(line: str):
    """Converts a line of made data to a dict"""
    items = re.split("\|", line)
    return {
        "id": items[0],
        "sentence": items[1],
        "entity1": items[2],
        "start1": items[3],
        "end1": items[4],
        "entity2": items[5],
        "start2": items[6],
        "end2": items[7]
    }


def convert_rel_to_brat(rel_file_path, type1, type2, relation):

    all_data = {}

    with open(rel_file_path, 'r') as f:
        rel_text = f.read()
        rel_text_lines = rel_text.split('\n')

    for line in rel_text_lines:
        if not is_valid_ade(line): continue
        d = line_to_dict(line)
        this_id = int(d["id"])

        if this_id not in all_data.keys():
            all_data[this_id] = [d]
        else:
            all_data[this_id].append(d)

    for this_id, data_list in all_data.items():
        global t, r_  # Must be global to maintain accuracy despite scoping

        # For the first input file, we know that T and R start at 1; however, for the second file,
        # we need the T and R to resume where we left off.
        if type1 == "AdverseDrugEvent":
            Entity.t = 1
            r_ = 1
        elif type1 == "Dose":
            with open(os.path.join(output_dir_name, str(this_id) + ".ann"), "r") as f:
                set2_text = f.read()

            set2_text_lines = set2_text.split("\n")

            for i in reversed(set2_text_lines):
                if i.startswith("R"):
                    # bob = re.findall("R\d+", i)[0]
                    # print(bob)
                    r_ = int(re.findall("R\d+", i)[0][1:]) + 1
                    break
            for i in reversed(set2_text_lines):
                if i.startswith("T"):
                    Entity.t = int(re.findall("T\d+", i)[0][1:]) + 1
                    break

            del set2_text_lines, set2_text

        output_text = ""
        these_entites = []
        for d in data_list:
            # output_text += convert_dict_to_line(d)
            ent1 = Entity(type1, int(d["start1"]), int(d["end1"]), d["entity1"])
            ent2 = Entity(type2, int(d["start2"]), int(d["end2"]), d["entity2"])

            these_entites = sorted(these_entites, key=lambda x: x.start)

            ent1_new = True
            ent2_new = True

            for e in these_entites:
                if ent1 == e:
                    ent1 = e
                    ent1_new = False
                    break
            for e in these_entites:
                if ent2 == e:
                    ent2 = e
                    ent2_new = False
                    break

            if ent1_new and ent2_new:
                ent1.add_t()
                these_entites.append(ent1)
                ent2.add_t()
                these_entites.append(ent2)
                output_text += str(ent1) + str(ent2)
            elif ent1_new and not ent2_new:
                ent1.add_t()
                these_entites.append(ent1)
                output_text += str(ent1)
            elif (not ent1_new) and ent2_new:
                ent2.add_t()
                these_entites.append(ent2)
                output_text += str(ent2)

            output_text += f"R{r_}\t{relation} Arg1:T{ent1.t} Arg2:T{ent2.t}\n"
            r_ += 1

        with open(os.path.join(output_dir_name, str(this_id) + ".ann"), "a+") as f:
            f.write(output_text)


if __name__ == "__main__":

    # Get the input and output directories from the command line.

    if len(argv) < 3:
        # Command-line arguments must be provided for the input and output directories.
        raise IOError("Please run the program again, entering the input and output directories as command-line"
                      " arguments in that order.")

    try:
        input_dir_name = argv[1]
        input_dir = os.listdir(input_dir_name)
    except FileNotFoundError:  # dir doesn't exist
        while not os.path.isdir(input_dir_name):
            input_dir_name = input("Input directory not found; please try another directory:")
        input_dir = os.listdir(input_dir_name)
    output_dir_name = argv[2]
    if not os.path.isdir(output_dir_name):
        raise FileNotFoundError("The provided output directory does not exist. Please create it and run this script again")

    convert_rel_to_brat(os.path.join(input_dir_name, "DRUG-AE.rel"), "AdverseDrugEvent", "Drug", "Drug-AdverseEvent")
    convert_rel_to_brat(os.path.join(input_dir_name, "DRUG-DOSE.rel"), "Dose", "Drug", "Amount")
