import duckdb
import pandas as pd

con = duckdb.connect("data/warehouse.duckdb")
tables = con.execute("PRAGMA show_tables").fetchall()

print("| Table | Status | All Columns |")
print("| :--- | :--- | :--- |")

for (tbl,) in tables:
    count = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({tbl})").fetchall()]
    status = f"{count:,} rows" if count > 0 else "Empty"
    # Join all columns with <br> or comma to fit in table, but user said "not every data opint", wait "thats not every data opint".
    # User wants every data point. I will list them all.
    col_str = ", ".join(f"`{c}`" for c in cols)
    print(f"| `{tbl}` | {status} | {col_str} |")
