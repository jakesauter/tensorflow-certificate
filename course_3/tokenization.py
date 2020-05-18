import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


sentences = []
labels = []

input_file = '../data/bbc-text.csv'

"""
wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv \
    -O /tmp/bbc-text.csv

"""

with open(input_file, 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	next(reader)
	for row in reader:
		labels.append(row[0])
		sentence = row[1]
		for word in stopwords:
			token = ' ' + word + ' '
			sentence = sentence.replace(token, ' ')
			sentence = sentence.replace('  ', ' ')
		sentences.append(sentence)

# now that sentences and labels are neatly in arrays, lets tokenize the sentences
# so that we can train on them 

# first we need to make an instance of a Tokenizer, and compose a word index
# with fit_on_texts

# oov = out of vocabulary
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# as seen below, a word index is a dictionary mapping
# of words to numbers that can be encoded for the DNN
print(word_index)
print(len(word_index))

# now that we have created a word index, lets encode the 
# sentences into sequences

sequences = tokenizer.texts_to_sequences(sentences)

# can also control the number of words encoded (most frequent words used)
sequences = tokenizer.texts_to_sequences(sentences, num_words = 100)


# now because all sentences are not the same length, we need to pad them
# we can choose to pad pre or post, and choose the max length, otherwise
# default is to pad pre at the length of the max length sequence

padded = pad_sequences(sequences, padding = 'post')

print(padded[0])
print(padded.shape)


## All together now
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences) # num words arg goes here
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding = 'post') 

# we can also be a bit more meticulous of how we would like to pad
padded = pad_sequences(sequences, padding = 'post',
			 			maxlen = 4, truncating = 'post')

# and if we want to decode model predictions for a sequence to sequence model
# 
# rev_word_index = dict({(value, key) for key,value in word_index.items()})
# 
# def decode(seq): 
# 	out = []
# 	for sym in seq: 
# 		val = rev_word_index
# 












