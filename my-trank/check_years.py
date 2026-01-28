
import csv
import sys

filename = 'season.csv'
years = set()

try:
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'torvik_year' in row and row['torvik_year']:
                years.add(row['torvik_year'])
            elif 'year' in row and row['year']:
                years.add(row['year'])
    
    print("Years found:", sorted(list(years)))

except Exception as e:
    print(e)
