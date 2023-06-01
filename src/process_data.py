import argparse
import sys

import pandas as pd

sys.path.append("/Users/harshitmishra/Documents/FEVEROUS")  # TODO
from pre_process.fetch_data_db import FetchData


class ProcessData:
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--train_path", type=str)
        parser.add_argument("--dev_path", type=str)
        # parser.add_argument("--train_path", type=str)
        args = parser.parse_args()
        db = FetchData("../data/feverous_wikiv1.db")

        def make_data_frame(path, db, name):
            df = pd.read_json(path, lines=True)
            df["text"] = ""

            n = len(df)
            for i in range(n):
                if "evidence" in df.loc[i].keys():
                    evidence = df.loc[i]["evidence"]
                    s = ""
                    for e in range(len(evidence)):
                        context = evidence[e]["context"]

                        for k in context.keys():
                            try:
                                k_page, k_key = k.split("_", 1)

                                s += str(db.fetch_text(k_page)[k_key]) + " "
                            except Exception as e:
                                continue

                        for v in context.values():
                            for v_i in range(len(v)):
                                try:
                                    v_page, v_key = v[v_i].split("_", 1)
                                    s += str(db.fetch_text(v_page)[v_key]) + " "
                                except Exception as e:
                                    continue
                    df.loc[i, "text"] = s
            df.to_csv(f"{name}.csv")

        make_data_frame(args.train_path, db, "augment_train_data")
        make_data_frame(args.dev_path, db, "augment_dev_data")
