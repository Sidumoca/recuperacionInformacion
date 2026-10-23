"""
index.py
Author: Javier Nogueras Iso
Last update: 2024-09-07

Simple program to create an inverted index with the contents of text/xml files contained in a docs folder
This program is based on the whoosh library. See https://pypi.org/project/Whoosh/ .
Usage: python index.py -index <index folder> -docs <docs folder>
"""

from whoosh.index import create_in
from whoosh.fields import *
from whoosh.analysis.filters import LowercaseFilter
from whoosh.analysis.tokenizers import RegexTokenizer
from whoosh.analysis.filters import StopFilter, STOP_WORDS
from whoosh.analysis.tokenizers import default_pattern
from nltk.stem.snowball import SnowballStemmer
from whoosh.analysis import Filter
import re
import sys
import os

import xml.etree.ElementTree as ET

def create_folder(folder_name):
    if (not os.path.exists(folder_name)):
        os.mkdir(folder_name)


class SnowballFilter(Filter):
    def __init__(self):
        self.stemmer = SnowballStemmer("spanish")

    def __call__(self, tokens):
        for t in tokens:
            t.text = self.stemmer.stem(t.text)
            yield t


def LanguageAnalyzerNuestro(lang, expression=default_pattern, gaps=False,
                     cachesize=50000):
    # Make the start of the chain
    return (RegexTokenizer(expression=expression, gaps=gaps)
             | LowercaseFilter() | StopFilter(lang=lang) | SnowballFilter())


class MyIndex:
    def __init__(self,index_folder):
        analyzer = LanguageAnalyzerNuestro(lang="es", expression=r"\w+")
        schema = Schema(
            path=ID(stored=True),
            identifier=ID(stored=True),
            autor=TEXT(analyzer=analyzer), 
            director=TEXT(analyzer=analyzer), 
            departamento=TEXT(analyzer=analyzer),
            title=TEXT(analyzer=analyzer),
            defenseYear=ID(stored=True),
            description=TEXT(analyzer=analyzer),
            subject=TEXT(analyzer=analyzer),
            type=ID(stored=True),
            language=ID(stored=True)
        )
        create_folder(index_folder)
        index = create_in(index_folder, schema)
        self.writer = index.writer()

    def index_docs(self,docs_folder):
        if (os.path.exists(docs_folder)):
            for file in sorted(os.listdir(docs_folder)):
                # print(file)
                if file.endswith('.xml'):
                    self.index_xml_doc(docs_folder, file)
                elif file.endswith('.txt'):
                    self.index_txt_doc(docs_folder, file)
        self.writer.commit()

    def index_txt_doc(self, foldername,filename):
        file_path = os.path.join(foldername, filename)
        # print(file_path)
        with open(file_path) as fp:
            text = ' '.join(line for line in fp if line)
        # print(text)
        self.writer.add_document(path=filename, content=text)

    def index_xml_doc(self, foldername, filename):
        file_path = os.path.join(foldername, filename)
        # print(file_path)
        try: #por si hay documento que pete a mitad que no se rompa todo
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Error procesando {filename}: {e}")
            return

        def get_tag_text(tag_name):
            #* es el comodín para saltarse el namespace
            elements = root.findall(f'.//{{*}}{tag_name}')
            #extrae el texto de todas las apariciones y descarta valores vacios
            texts = [elem.text.strip() for elem in elements if elem.text]
            return ' '.join(texts)

        def get_first_tag_text(tag_name):
            elements = root.findall(f'.//{{*}}{tag_name}')
            for element in elements:
                if element.text and element.text.strip():
                    return element.text.strip()
            return ''

        title_text = get_tag_text('title')
        autor_text = get_tag_text('creator')
        description_text = get_tag_text('description')
        director_text = get_tag_text('contributor')
        departamento_text = get_tag_text('publisher')

        raw_date = get_tag_text('date')
        match = re.search(r'\d{4}', raw_date)
        year_text = match.group(0) if match else ''

        subject_text = get_tag_text('subject')
        identifier_text = get_first_tag_text('identifier')
        type_text = get_first_tag_text('type')
        # si hay varios idiomas, se pone el primero
        language_text = get_first_tag_text('language')
            
        self.writer.add_document(
            path=filename,
            identifier=identifier_text,
            title=title_text,
            autor=autor_text,
            director=director_text,
            departamento=departamento_text,
            subject=subject_text,
            description=description_text,
            defenseYear=year_text,
            type=type_text, 
            language=language_text
        )
if __name__ == '__main__':

    index_folder = '../whooshindex'
    docs_folder = '../docs'
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-index':
            index_folder = sys.argv[i + 1]
            i = i + 1
        elif sys.argv[i] == '-docs':
            docs_folder = sys.argv[i + 1]
            i = i + 1
        i = i + 1

    my_index = MyIndex(index_folder)
    my_index.index_docs(docs_folder)
