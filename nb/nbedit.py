# nb.py - naive bayes classifier in form P(category|words) = P(words|category) * P(category)
import re
import math
import os
import string

def sanitize(text):
	table = string.maketrans("","")
	text = text.translate(table, string.punctuation)
	text = text.lower()
	return text
	
# function to create a dictionary of word counts from a text
def make_word_count(text):
	word_count = {}
	for word in re.split("\W+", text):
#		if len(word) <= 3:
#			continue
		word_count[word] = word_count.get(word, 0.0) + 1.0
	return word_count

vocab = {}
p_word_given_category = {}
p_category = {}

# get categories from directories in ./samples/
def make_probabilities(directory = './samples/'):
	category_word_counts = {}
	for category in os.listdir(directory):
		category_word_counts[category] = {}
		p_word_given_category[category] = {}
		p_category[category] = 0.0
		# make word count of documents in ./samples/*category_directory*
		for doc in os.listdir(directory + category):
			counts = make_word_count(open(directory + category + '/' + doc).read())
			p_category[category] += 1
			# add words to the vocab and category_word_counts
			for word, count in counts.items():
				if word not in vocab:
					vocab[word] = 0.0
				if word not in category_word_counts[category]:
					category_word_counts[category][word] = 0.0
				vocab[word] += count
				category_word_counts[category][word] += count
		#calculate P(word|category for each word in category
		for word, count in category_word_counts[category].iteritems():
			p_word_given_category[category][word] = category_word_counts[category].get(word, 0.0) / sum(category_word_counts[category].values())
	# finish calculating p_category
	numsdoc = sum(p_category.values())
	for category in p_category:
		p_category[category] = p_category[category] / numsdoc

# return 'maximum likelihood' category given text
def get_category(s):
	# initialize dictionaries
	p_category_given_words = {}
	for category in p_word_given_category:
		p_category_given_words[category] = 0.0
	
	# for every word in our test sentence, make P(word|category) for each category,
	# compound them and then add P(category) to give P(category|words in test sentence)
	for word, count in make_word_count(s).items():
		# skip word if not in vocab
		if not word in vocab:
			continue
		for category in p_word_given_category:
			# add (log) probabilities to our running count, giving probability given *category*
			if p_word_given_category[category].get(word, 0.0) > 0.0:
				# note: x * log(z) == log(z^x), i.e. a word repeated in the sample counts as several features
				p_category_given_words[category] += count * math.log(p_word_given_category[category][word])
	# add p_category and then convert probabilities from log
	for category in p_category_given_words:
		p_category_given_words[category] = math.exp(p_category_given_words[category]+math.log(p_category[category]))
	# still need to add max function
	return p_category_given_words
	


