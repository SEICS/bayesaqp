import pickle
import pandas as pd

true_card = pd.read_csv("true_cards_joblightranges.csv")
col_name = true_card.columns[0]
true_card = pd.to_numeric(true_card[col_name]).to_list()
card_dict = {i: [tuple([tc])] for i, tc in enumerate(true_card)}

with open("imdb_job_light_ground_truth.pkl", "wb") as f:
    pickle.dump(card_dict, f)