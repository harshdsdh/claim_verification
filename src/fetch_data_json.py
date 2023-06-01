import sys

import pandas as pd

sys.path.append("/Users/harshitmishra/Documents/FEVEROUS")  # TODO
from pre_process.fetch_data_db import FetchData

if __name__ == "__main__":
    db = FetchData("../data/feverous_wikiv1.db")

    df = pd.read_json("../data/train.jsonl", lines=True)
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
            print(s)
            df.loc[i, "text"] = s
    df.to_csv("augmented_data.csv")
