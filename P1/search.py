"""
search.py
Author: Javier Nogueras Iso
Last update: 2024-09-07

Program to search a free text query on a previously created inverted index.
This program is based on the whoosh library. See https://pypi.org/project/Whoosh/ .
Usage: python search.py -index <index folder>
"""

import sys

from whoosh.qparser import QueryParser
from whoosh.qparser import OrGroup
from whoosh import scoring
import whoosh.index as index
from nltk.stem.snowball import SnowballStemmer
from whoosh.analysis import Filter

class SnowballFilter(Filter):
    def __init__(self):
        self.stemmer = SnowballStemmer("spanish")

    def __call__(self, tokens):
        for t in tokens:
           t.text = self.stemmer.stem(t.text)
           yield t

class MySearcher:
    def __init__(self, index_folder, model_type = 'tfidf'):
        ix = index.open_dir(index_folder)
        if model_type == 'tfidf':
            # Apply a vector retrieval model as default
            self.searcher = ix.searcher(weighting=scoring.TF_IDF())
        else:
            # Apply the probabilistic BM25F model, the default model in searcher method
            self.searcher = ix.searcher()
        self.parser = QueryParser("content", ix.schema, group = OrGroup)

    def search(self, query_text, query_index, output):
        query = self.parser.parse(query_text)
        results = self.searcher.search(query, limit=100) #limite 100
        i = 1
        for result in results:
            ident = result.get("identifier")
            if ident:
                print(f'{query_index}\t{ident}', file=output)
            print(f'{query_index}.{i} - File path: {result.get("path")}, Similarity score: {result.score}')
            i += 1

if __name__ == '__main__':
    index_folder = '../whooshindex'
    query_file = '../queries'
    results_file = '../whooshresults'
    i = 1
    while (i < len(sys.argv)):
        if sys.argv[i] == '-index':
            index_folder = sys.argv[i+1]
            i = i + 1
        elif sys.argv[i] == '-infoNeeds':
            query_file = sys.argv[i+1]
            i = i + 1
        elif sys.argv[i] == '-output':
            results_file = sys.argv[i+1]
            i = i + 1
        i = i + 1

    searcher = MySearcher(index_folder)

    with open(query_file, 'r') as queries, open(results_file, 'w') as output:
        for query_index, query in enumerate(queries, 1):
            query = query.strip()
            if query:
                searcher.search(query, query_index, output)