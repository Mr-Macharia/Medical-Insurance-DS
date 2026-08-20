"""Cell constructors shared by the section modules."""

import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text.strip('\n'))


def code(text):
    return nbf.v4.new_code_cell(text.strip('\n'))
