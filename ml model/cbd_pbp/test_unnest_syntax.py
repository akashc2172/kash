stop
import duckdb
import pandas as pd

def test_syntax():
    con = duckdb.connect(':memory:')
    
    # Create table with struct array
    con.execute("CREATE TABLE play (id INTEGER, onFloor STRUCT(id INTEGER, name VARCHAR, team VARCHAR)[])")
    
    # Insert data
    # DuckDB python client can insert list of dicts
    data = [
        (1, [{'id': 101, 'name': 'Player A', 'team': 'Duke'}, {'id': 102, 'name': 'Player B', 'team': 'UNC'}])
    ]
    con.execute("INSERT INTO play VALUES (?, ?)", data[0])
    
    print("Data inserted.")
    
    queries = [
        # 1. Alias columns in UNNEST
        "SELECT t.id, t.name FROM play, UNNEST(onFloor) as t(id, name, team)",
        # 2. Access struct field via dot on unnest result directly
        "SELECT (unnest(onFloor)).id, (unnest(onFloor)).name FROM play",
        # 3. Access struct field via map syntax on alias
        "SELECT t['id'], t['name'] FROM play, UNNEST(onFloor) as t",
        # 4. Simple unnest without alias? 
        "SELECT unnest(onFloor).id FROM play"
    ]

    for i, q in enumerate(queries):
        print(f"\\n--- Trying Query {i+1} ---")
        try:
            df = con.execute(q).df()
            print("Success!")
            print(df)
        except Exception as e:
            print(f"Failed: {e}")
        
    con.close()

if __name__ == "__main__":
    test_syntax()
