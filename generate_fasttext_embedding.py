# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import argparse
import sys
import time
import warnings

from flair.embeddings import FastTextEmbeddings 
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
from tqdm import tqdm

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-file", required=True, help="File containing the words")

    args = vars(ap.parse_args())
    edges_file = args["vocab_file"]

    edges_path = edges_file.split("/")
    edges_filename = edges_path[-1].split(".")[0]
    embeddings_file = "/".join(edges_path[:-1]) + "/" + edges_filename + "_fasttext_embeddings.csv"

    words = []
    with open(edges_file, "r") as f:
        for line in f:
            word = " ".join(line.strip().split("_")) # space-separate phrases
            words.append(word)
    sentences = [Sentence(word, use_tokenizer=True) for word in words]

    t0 = time.time()
    embedding = FastTextEmbeddings("crawl-300d-2M-subword.bin")
    doc_embedding = DocumentPoolEmbeddings([embedding]) # doc-embedding via mean-pooling
    print("Loading fasttext model: " + str(time.time() - t0) + "s")

    doc_embedding.embed(sentences)

    word_embeddings = []
    for sentence in sentences:
        word_embeddings.append(sentence.get_embedding())
    
    print("length: " + str(len(word_embeddings[0])))

    with open(embeddings_file, "w") as f:
        for word, word_embedding in zip(words, word_embeddings):
            f.write("_".join(word.split(" ")) + "," +\
                    ",".join(["{:0.32f}".format(e) for e in word_embedding]) +\
                    "\n")

    print("Wrote embeddings to: " + embeddings_file)
