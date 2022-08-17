import csv
import json
import pandas as pd
from tqdm import tqdm

with open("datasets/en_jp/test.txt", "r", encoding="utf-8") as file:
    jps = []
    ens = []
 
    for line in tqdm(file):
        line = line.replace("\n", "").replace("\r\n", "")
        
        try:
            en_text, jp_text = line.split("|")
        except Exception:
            en_text, jp_text = line.split("\t")
            
        ens.append(en_text)
        jps.append(jp_text)
    
    raw_data = {"english": ens, "japanese": jps}
    df = pd.DataFrame(raw_data, columns=["english", "japanese"])

    df.to_csv("datasets/en_jp/test.csv", index=False)



            
