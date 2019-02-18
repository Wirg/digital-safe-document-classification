import os
import subprocess


class ConvertFileException(Exception):
    pass


EXTENSION_ALIASES = {
    'text': ['txt', 'text', 'md'],
    'pdf': ['pdf']
}


FILE_READERS = {
    'pdf': lambda content: subprocess.check_output([f'pdftotext', "-", "-"], input=content).decode('utf-8'),
    'text': lambda content: content.decode("utf-8", 'backslashreplace')
}


def get_file_type(filename):
    _, extension = os.path.splitext(filename.lower())
    for extension_type, extensions in EXTENSION_ALIASES.items():
        if extension[1:] in extensions:
            return extension_type

    raise ConvertFileException(f'Unknown file extension {extension} for file {filename}')


def read_file(file):
    file_type = get_file_type(file.filename)

    if file_type not in FILE_READERS:
        raise ConvertFileException(f'No text converter for file_type {file_type}')

    return FILE_READERS[file_type](file.read())
