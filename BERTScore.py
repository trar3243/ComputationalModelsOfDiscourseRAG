#pip install bert-score pandas numpy
from bert_score import score
import json
import pandas as pd
import numpy as np

"""
INSPECT LOONG DATASET JSON
"""
with open("loong.jsonl", "r", encoding="utf-8") as f:
    obj = json.loads(next(f))

#PRINT KEYS
print(obj)
print("\nKeys:", obj.keys())

#INSTANTIATE LIST FOR GROUND TRUTHS
truth_rows = []

"""
LOAD REFERENCE ANSWERS
"""
#LOAD IN LOONG
with open("loong.jsonl", "r", encoding="utf-8") as f:
    #PER LINE PULL ANSWER
    for line in f:
        obj = json.loads(line)
        ans = obj["answer"]
        #GET GROUND TRUTH REFERENCE IF IT EXISTS, ADD TO LIST
        if isinstance(ans, dict):
            ref_list = ans.get("Reference", [])
        #ELSE NONE
        else:
            ref_list = []
        #FLATTEN LIST TO STRING
        ref_text = " ".join(ref_list)
        #APPEND TO ABOVE GROUND TRUTH LIST: QUESTION, REFERENCE TEXT, LEVEL
        truth_rows.append({
            "question": obj["question"],
            "reference": ref_text,
            "level": obj.get("level")
        })
#CONVERT TO DATAFRAME
df_truth = pd.DataFrame(truth_rows)

"""
LOAD COREFRAG ANSWERS
"""
#INSTANTIATE LIST FOR COREFRAG OUTPUT
rag_rows = []

#LOAD IN COREFRAG OUTPUT
with open("rag_outputs.jsonl", "r", encoding="utf-8") as f:
    #PER LINE PULL ANSWER
    for line in f:
        obj = json.loads(line)
        ans = obj["answer"]
        #GET COREFRAG ANSWER IF IT EXISTS, ADD TO LIST
        if isinstance(ans, dict):
            ref_list = ans.get("Reference", [])
        #ELSE NONE
        else:
            ref_list = []
        #FLATTEN LIST TO STRING
        ref_text = " ".join(ref_list)
        #APPEND TO ABOVE COREFRAG ANSWER: QUESTION, REFERENCE TEXT, LEVEL
        rag_rows.append({
            "question": obj["question"],
            "prediction": ref_text,
            "level": obj.get("level")
        })
#CONVERT TO DATAFRAME
df_rag = pd.DataFrame(rag_rows)

"""
MERGE, RUN BERTSCORE
"""

#ALIGN BY QUESTION
df_all = df_rag.merge(df_truth, on=["question", "level"], how="inner")

#CHECK DATAFRAME WITH ALL
print(len(df_all))
df_all.head()

#INSTANTIATE LISTS, RUN BERTSCORE
references = df_all["reference"].tolist()
predictions = df_all["prediction"].tolist()

P, R, F1 = score(
    cands=predictions,
    refs=references,
    lang="en",
    model_type="roberta-large",
    verbose=True
)

#GET, STORE SCORES, PRINT AVERAGE OVER RESULTS
df_all["bertscore_precision"] = P.tolist()
df_all["bertscore_recall"] = R.tolist()
df_all["bertscore_f1"] = F1.tolist()

print("Overall BERTScore F1:", df_all["bertscore_f1"].mean())
