import gensim
import logging  # logging
import argparse  # handle params
from time import time  # To time our operations
import pandas as pd  # For data handling

logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# df = pd.read_csv("data/simpsons_dataset.csv")
# print(df.shape)
# print(df.head)

# df = df.dropna().reset_index(drop=True)
# print(df.isnull().sum())

# t = time()

# # simple pre processing of data
# documents = [gensim.utils.simple_preprocess(row) for row in df["spoken_words"]]

# print("Time to clean up everything: {} mins".format(round((time() - t) / 60, 2)))

# # build vocabulary and train model
# model = gensim.models.Word2Vec(documents, window=10, min_count=2, workers=10)

# print("Time to train the model: {} mins".format(round((time() - t) / 60, 2)))

# # w2v_model.init_sims(replace=True)

# print("words similar to homer", model.wv.most_similar(positive=["homer"]))

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process a data file and column name.")
    parser.add_argument("filepath", help="Path to the data file")
    parser.add_argument("term", help="Search term to find similar items to")
    parser.add_argument(
        "--col", default=None, help="Name of the column to be processed"
    )
    parser.add_argument(
        "--show-df", action="store_true", help="Show the kg data frame and exit"
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Read the data
    data = pd.read_csv(args.filepath)
    # df = pd.read_csv("data/simpsons_dataset.csv")

    search_term = args.term

    # Use the first column if --col is not provided
    col = args.col if args.col is not None else data.columns[0]

    data_text = data[[col]].copy()  # create a copy to avoid warnings

    # Ensure all text data are string type
    data_text[args.col] = data_text[col].astype(str)

    # simple clean of the data dropping nulls
    t = time()
    data_text = data_text.dropna().reset_index(drop=True)
    # simple pre processing of data
    documents = [gensim.utils.simple_preprocess(row) for row in data_text[col]]

    print("Time to clean up everything: {} mins".format(round((time() - t) / 60, 2)))

    # build vocabulary and train model
    model = gensim.models.Word2Vec(documents, window=10, min_count=2, workers=10)

    print("Time to train the model: {} mins".format(round((time() - t) / 60, 2)))

    similar_docs = model.wv.most_similar(positive=[search_term])

    print("words similar to ", search_term, similar_docs)
