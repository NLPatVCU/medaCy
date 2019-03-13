"""
:author: Steele Farnsworth
:date: 13 March, 2019
"""


class Line:
    """
    Represents a line of text in the text file related to an annotation file, ensuring that each line has an accurate
    start index as one of its attributes regardless of whether that line appears more than once
    """

    def __init__(self, line_text: str, line_num: int, line_index: int):
        self.text = line_text
        self.num = line_num
        self.index = line_index

    @staticmethod
    def init_lines(full_text: str):
        """
        Creates all the Line objects for a given text file, storing them in a list where index n is the nth - 1
        line of the document.
        :param full_text: The entire text of the document.
        :return: The list of Lines.
        """
        global_start_ind = 0
        global_line_num = 0

        full_text_lines = full_text.split('\n')
        text_lines = []

        for given_line in full_text_lines:

            sub_index = 0
            matches = []
            while sub_index < global_start_ind:
                for previous_line in text_lines:
                    if given_line == previous_line.text:
                        matches.append(previous_line)
                    sub_index += previous_line.index

            if matches:
                # Get the text from the end of the last match onward
                search_text_start = matches[-1].index + len(matches[-1].text)
                search_text = full_text[search_text_start:]
                start_ind = search_text.index(given_line) + search_text_start
            else:  # The line is unique so str.index() will be accurate
                start_ind = full_text.index(given_line)

            new_line = Line(given_line, global_line_num, start_ind)
            text_lines.append(new_line)

            global_start_ind = text_lines[-1].index
            global_line_num += 1

        return text_lines

    def __str__(self):
        """String representation of a line, with its index and text separated by a pipe."""
        return "%i | %s" % (self.index, self.text)
