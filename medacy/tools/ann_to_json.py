"""
Converts ann file text to spacy annotation
"""

def ann_to_json(annotation_text):
    annotations = {'entities': {}, 'relations': []}

    for line in annotation_text.split("\n"):
        if "\t" in line:
            line = line.split("\t")
            if 'T' in line[0]:
                tags = line[1].split(" ")
                entity_name = tags[0]
                entity_start = int(tags[1])
                entity_end = int(tags[-1])
                annotations['entities'][line[0]] = (entity_start, entity_end, entity_name)

            if 'R' in line[0]:
                tags = line[1].split(" ")
                relation_name = tags[0]
                relation_start = tags[1].split(':')[1]
                relation_end = tags[2].split(':')[1]
                annotations['relations'].append((relation_name, relation_start, relation_end))

    return annotations
