import sqlite3
import csv
import pdb
from tqdm import tqdm
conn = sqlite3.connect('./data/retrieval/colbertv2.0_passages/wiki2023/enwiki-20230401.db')
print('load success!')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [table[0] for table in cursor.fetchall()]
tsv_data = []
for table_name in tables: 
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    # columns = [desc[0] for desc in cursor.description] # ['title', 'text']
    
    for id,row in tqdm(enumerate(rows), total=len(rows), desc=f"Processing {table_name}"): 
        title = row[0]
        text = row[1]
        id = str(id)
        data = id + '\t' + text + '\t' + title +'\n'
        tsv_data.append(data)
# build TSV test tsv file
tsv_file = "./data/retrieval/colbertv2.0_passages/wiki2023/enwiki-20230401.tsv"
with open(tsv_file, "w") as file:
    file.writelines(tsv_data)
    print(f"{table_name} exported to {tsv_file}")

cursor.close()
conn.close()
