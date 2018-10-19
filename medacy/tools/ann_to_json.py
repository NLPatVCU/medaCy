"""
Converts ann file text to spacy annotation
"""

def ann_to_json(annotation_text):
    annotations = {'entities': []}

    for line in annotation_text.split("\n"):
        if "\t" in line:
            line = line.split("\t")
            if 'T' in line[0]:
                tags = line[1].split(" ")
                entity_name = tags[0]
                entity_start = int(tags[1])
                entity_end = int(tags[-1])
                annotations['entities'].append((entity_start, entity_end, entity_name))

    return annotations

