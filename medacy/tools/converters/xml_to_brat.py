# from sys import argv
# import os
# import re
# import logging
# from bs4 import BeautifulSoup
#
#
# def switch_extension(name, ext):
#     """Takes the name of a file (str) and changes the extension to the one provided (str)"""
#     return os.path.splitext(name)[0] + ext
#
#
# def convert_xml_to_brat(xml_file_path):
#     with open(xml_file_path) as f:
#         xml_text = f.read()
#
#     xml_text = re.sub("&gt;", ">", xml_text)
#     whole_soup = BeautifulSoup(xml_text, features="html.parser")
#     abstract_in_tags = whole_soup.find("abstracttext")
#     abstract_soup = BeautifulSoup(str(abstract_in_tags), features="html.parser")
#
#     # The text of the abstract without any tags
#     abstract_text = abstract_soup.get_text()
#
#     tags = set([tag.name for tag in abstract_soup.find_all()])
#     tags.remove("abstracttext")
#
#     infix = "( |<.*?>|)+"
#     circumfix = "<.*?>"
#
#     all_annotations = []
#
#     for tag in tags:
#
#         all_for_tag = abstract_soup.find_all(tag)
#
#         escaped_patterns = []
#         for tagged_item in all_for_tag:
#             tag_pattern = re.escape(str(tagged_item))
#             if tag_pattern not in escaped_patterns:
#                 escaped_patterns.append(tag_pattern)
#
#         matches_for_tag = []
#         for pattern in escaped_patterns:
#             iter_matches = re.finditer(pattern, str(abstract_soup))
#             for match in iter_matches:
#                 matches_for_tag.append(match)
#
#         # Get all the instances of that tag
#         for tagged_item in matches_for_tag:
#             # We only want to search up to the end of the match we're looking at
#             cap_index = tagged_item.span()[1]
#             match_text = tagged_item.string[tagged_item.span()[0]:tagged_item.span()[1]]
#             match_soup = BeautifulSoup(match_text, features="html.parser")
#             match_tagless = match_soup.get_text()
#
#             # Construct the regex pattern to get all instances of the phrase
#             # in the xml file (not the tagless version)
#             # regardless of whether there are tags in between words
#             tagless_escaped = re.escape(match_tagless)
#             spaced = re.sub(r"\\ ", infix, tagless_escaped)
#             spaced = re.sub(r"\\-", infix + "-" + infix, spaced)
#             spaced = re.sub(r"\\,", r"," + infix, spaced)
#             spaced = re.sub(r"\\/", infix + r"\\/" + infix, spaced)
#             circumfixed = circumfix + spaced + circumfix
#
#             # Figure out how many matches come before the instance we're looking at, including itself
#             search_text = str(abstract_soup)[:cap_index]
#             search_text = re.sub("&gt;", ">", search_text)
#             similar_matches = list(re.finditer(circumfixed, search_text))
#             specific_instance = len(similar_matches) - 1
#             assert specific_instance >= 0, "specific_instance should never be negative"
#
#             # Find the same instance of the entity in the txt version
#             parallel_matches = list(re.finditer(spaced, abstract_text))
#             specific_match = parallel_matches[specific_instance]
#
#             new_annotation = {
#                     "entity": specific_match.string[specific_match.span()[0]:specific_match.span()[1]],
#                     "entity_type": tag,
#                     "start_ind": specific_match.span()[0],
#                     "end_ind": specific_match.span()[1]
#                 }
#
#             all_annotations.append(new_annotation)
#
#     all_annotations = sorted(all_annotations, key=lambda x: (x["start_ind"], x["end_ind"]))
#
#     brat_text = ""
#     t = 1
#
#     for d in all_annotations:
#         brat_line = "T%i\t%s %i %i\t%s\n" % (t, d["entity_type"], d["start_ind"], d["end_ind"], d["entity"])
#         brat_text += brat_line
#         t += 1
#
#     return brat_text, abstract_text
#
#
# if __name__ == "__main__":
#     # Get the input and output directories from the command line.
#
#     if len(argv) < 3:
#         # Command-line arguments must be provided for the input and output directories.
#         raise IOError("Please run the program again, entering the input and output directories as command-line"
#                       " arguments in that order.")
#
#     try:
#         input_dir_name = argv[1]
#         input_dir = os.listdir(input_dir_name)
#     except FileNotFoundError:  # dir doesn't exist
#         while not os.path.isdir(input_dir_name):
#             input_dir_name = input("Input directory not found; please try another directory:")
#         input_dir = os.listdir(input_dir_name)
#     try:
#         output_dir_name = argv[2]
#         output_dir = os.listdir(output_dir_name)
#     except FileNotFoundError:
#         while not os.path.isdir(output_dir_name):
#             output_dir_name = input("Output directory not found; please try another directory:")
#             output_dir = os.listdir(output_dir_name)
#
#     # Create the log
#     log_path = os.path.join(output_dir_name, "conversion.log")
#     logging.basicConfig(filename=log_path)
#
#     # Get only the text files in input_dir
#     xml_files = [f for f in input_dir if f.endswith(".xml")]
#
#     # Ensure user is aware if there are no files to convert
#     if len(xml_files) < 1:
#         raise FileNotFoundError("There were no xml files in the input directory.")
#
#     for input_file_name in xml_files:
#         full_file_path = os.path.join(input_dir_name, input_file_name)
#         ann_str, txt_str = convert_xml_to_brat(full_file_path)
#
#         output_ann_name = switch_extension(input_file_name, ".ann")
#         output_ann_path = os.path.join(output_dir_name, output_ann_name)
#         with open(output_ann_path, "a+") as f:
#             f.write(ann_str)
#
#         output_txt_name = switch_extension(input_file_name, ".txt")
#         output_txt_path = os.path.join(output_dir_name, output_txt_name)
#         with open(output_txt_path, "a+") as f:
#             f.write(txt_str)
