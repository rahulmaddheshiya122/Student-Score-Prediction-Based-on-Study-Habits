import sqlite3, pandas as pd, sys

if len(sys.argv) < 3:
    print("Usage: python sql_setup.py <csv_path> <db_name.sqlite> [table_name]")
    sys.exit(1)

csv_path = sys.argv[1]
db_name = sys.argv[2]
table = sys.argv[3] if len(sys.argv) > 3 else "scores"

df = pd.read_csv(csv_path)
conn = sqlite3.connect(db_name)
df.to_sql(table, conn, if_exists="replace", index=False)
conn.close()
print(f"Imported {len(df)} rows from {csv_path} into {db_name}:{table}")