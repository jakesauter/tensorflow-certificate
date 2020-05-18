import json

json_file = '../data/sarcasm.json'

datastore = []

with open(json_file, 'r') as f: 
    for line in f:
	    datastore.append(json.loads(line))

sentences = []
labels = []
urls = []

for item in datastore: 
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])


