import json

from .unic2ascii import UNICODE_TO_ASCII


def convert(text):
    """Takes in a text string and converts it to ASCII,
    keeping track of each character change

    Arguments:
        text: string
            - The text to be converted
    
    Returns:
        text: string
            - The converted text
        diff: list of dicts with keys 'start', 'length', 'original'
            - Record of all ASCII conversions, each entry
              records the original non-ASCII character and
              the starting index and length of the replacement
    """
    diff = list()
    offset = 0
    for i, char in enumerate(text):
        if char in UNICODE_TO_ASCII and UNICODE_TO_ASCII[char] is not char:
            ascii = UNICODE_TO_ASCII[char]
            text = text[:i+offset] + ascii + text[i+1+offset:]
            diff.append({
                'start': i+offset,
                'length': len(ascii),
                'original': char
            })
            offset += len(ascii) - len(char)
    return text, diff


def restore(text, diff, metamap_dict):
    """Takes in non-ascii text and the list of changes made to it from the `convert()` function,
    as well as a dictionary of metamap taggings, converts the text back to its original state
    and updates the character spans in the metamap dict to match

    Arguments:
        text, diff: string, list of dicts with keys 'start', 'length', 'original'
            - Outputs of `convert()` function
        metamap_dict: dictionary
            - Dictionary of metamap information obtained from text
    
    Returns:
        text: string
            - The input text with all of the changes listed in diff undone
        metamap_dict: dictionary
            - The input metamap_dict with all of its character spans updated to
                match the changes to the text
    """
    offset = 0
    for conv in diff: # Go through each recorded change to undo it & update metamap character spans accordingly
        conv_start = conv['start'] + offset
        conv_end = conv_start + conv['length']-1 # Ending index of converted span, INCLUSIVE

        # Undo the change to the text (restore ascii characters)
        text = text[:conv_start] + conv['original'] + text[conv_end+1:]
        delta = len(conv['original']) - conv['length']
        offset += delta

        # Check each metamap entry and update its character spans to reflect this change
        for mapping in metamap_dict['metamap']['MMOs']['MMO']['Utterances']['Utterance']['Phrases']['Phrase']['Mappings']['Mapping']:
            for candidate in mapping['MappingCandidates']['Candidate']:
                match_start = int(candidate['ConceptPIs']['ConceptPI']['StartPos'])
                match_length = int(candidate['ConceptPIs']['ConceptPI']['Length'])
                match_end = match_start + match_length-1

                if match_start == conv_start and match_end == conv_end: # If match is equal to conversion (a [conversion] and some text)
                    # print("Perfect match")
                    match_length += delta
                elif match_start < conv_start and match_end < conv_end: # If match intersects conversion on left ([a con]version and some text)
                    # print("Left intersect")
                    match_length += delta + conv_start
                elif conv_start < match_start and conv_end < match_end: # If match intersects conversion on right (a conver[sion and som]e text)
                    # print("Right intersect ")
                    if conv_end + delta < match_start:
                        print(match_end, conv_end)
                        match_start = conv_end + delta + 1
                        match_length = match_end - conv_end
                    else:
                        match_length += delta
                elif conv_end < match_start: # If match is totally to the right of the conversion (a conversion and a [match])
                    # print("Full right")
                    match_start += delta
                else: # If match is totally to right of conversion, no action needed (a [match] and a conversion)
                    # print("Full left")
                    pass

                # Update metamap entry with new indices
                candidate['MatchedWords']['MatchedWord'] = text[match_start:match_end+1]
                candidate['ConceptPIs']['ConceptPI']['StartPos'] = str(match_start)
                candidate['ConceptPIs']['ConceptPI']['Length'] = str(match_length)
    return text, metamap_dict
