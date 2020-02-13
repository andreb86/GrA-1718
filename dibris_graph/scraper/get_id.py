import requests
import logging
from bs4 import BeautifulSoup
import sys
import logging
import csv
import scholarly

'''
Scraping user id from scholar
file1 = open('../sorted_data.txt', 'r') 
Lines = file1.readlines() 
  

for line in Lines: 
    r = requests.get('https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=' + line.replace(", ", " ")[:-1] + '+"unige"+OR+"genova"+OR+"genoa"&btnG=')
    soup = BeautifulSoup(r.text, features="html.parser")

    # Check if multi-name:
    links = soup.findAll("a", {"class": "gs_ai_pho"})
    if len(links) > 1:
        print("### User: " + line.replace(", ", " ")[:-1] + "do it by hand", file=sys.stderr, flush=True)
    elif len(links) == 1:
        userid = links[0].get("href")[22:]
        print(userid + ", " + line[:-1], flush=True)
    elif len(links) == 0:
        print("User: " + line.replace(", ", " ")[:-1] + " not found", file=sys.stderr, flush=True)
'''

''' 
Scraping connection
'''
# Lista dell adiacenze su cui vado a scrivere
adj_list = open("adjacent_list.txt", "r+")


# Prendo tutti gli autori
autori = dict()
with open('../user_id.txt', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    # Per ogni user
    for row in csv_reader:
        name = row['name'] + " " + row['surname']
        autori[name] = row['id']

# Prendo gli autori già completati
completed = []
file_completati = open('completed', 'r+') 
Lines = file_completati.readlines() 
for line in Lines:
    completed.append(line[:-1])

for id_autore in autori.values():
    print("Inizio l'user: ", id_autore)
    if id_autore in completed:
        print("L'user: ", id_autore, " già fatto")
        continue
    print(id_autore, end=" ", file=adj_list)
    a = scholarly.Author(id_autore)
    a.fill()
    i = 0
    for pubblicazione in a.publications:
        print("Sono alla publicazione: ", i , " di ", len(a.publications))
        pubblicazione.fill()
        # AA Potrebbe essere un brevetto, per cui non ha autori
        try: 
            autori_articolo = pubblicazione.bib['author']
        except KeyError as ke:
            continue
        # Per ogni autore cerco se è dibris e lo aggiungo nella lista delle adiacenze
        array = autori_articolo.split(" and ")
        for nome_autore in array:
            try:
                author_id = autori[nome_autore]
                if author_id != id_autore:
                    print(autori[nome_autore], end=" ", file=adj_list)
            except KeyError as ke:
                pass
        i += 1
    print("", file=adj_list, flush=True)
    print(id_autore, file=file_completati, flush=True)