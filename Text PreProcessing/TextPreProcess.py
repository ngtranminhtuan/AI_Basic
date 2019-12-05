# 0. CSV load 
import pandas as pd
data = pd.read_csv('IMDB_Dataset.csv')
data.head()

# 1. Convert text to lowercase

input_str = ”The 5 biggest countries by population in 2017 are China, India, United States, Indonesia, and Brazil.”
input_str = input_str.lower()
print(input_str)
# Output:
# the 5 biggest countries by population in 2017 are china, india, united states, indonesia, and brazil.


# 2. Numbers removing
import re

input_str = ’Box A contains 3 red and 5 white balls, while Box B contains 4 red and 2 blue balls.’
result = re.sub(r’\d+’, ‘’, input_str)
print(result)

# Output:
# Box A contains red and white balls, while Box B contains red and blue balls.

# 3. Punctuation removal

import re
# Import các thư viện cần thiết
import csv
import itertools
import operator
import numpy as np
# Sử dụng thư viện nltk (Natural Language Toolkit) để phân tách dữ liệu thô
import nltk
nltk.download('punkt')
nltk.download("book")

def clean_data(text):
    text = re.sub("[^a-zA-Z\s]", "", text)
    text = re.sub("\s+", " ", text)
    return text
	
cleaned_data = []

for review in data.review:
    cleaned_review = clean_data(review)
    cleaned_data.append(cleaned_review)
	
print(cleaned_data[3])

# 4. White spaces removal

input_str = “ \t a string example\t “
input_str = input_str.strip()
input_str

# Output:
#‘a string example’

# 5. Tokenization
	# is the process of splitting the given text into smaller pieces called tokens. 
	# Words, numbers, punctuation marks, and others can be considered as tokens. 
	# In this table (“Tokenization” sheet) several tools for implementing tokenization are described.

vocabulary_size = 16000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
    
# Phân tách câu thành các từ
tokenized_sentences = [nltk.word_tokenize(sent) for sent in cleaned_data]

# Đếm tần suất xuất hiện của từ
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# Tìm ra các từ phổ biến nhất, xây dựng bộ từ điển
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Thay thế các từ không nằm trong từ điển bởi `unknown token`, lưu kết quả tiền xử lý câu
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: '%s'" % cleaned_data[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

# 6. Stop words removal
	# “Stop words” are the most common words in a language like “the”, “a”, “on”, “is”, “all”. 
	# These words do not carry important meaning and are usually removed from texts. 
	# It is possible to remove stop words using Natural Language Toolkit (NLTK), 
	# a suite of libraries and programs for symbolic and statistical natural language processing.

input_str = “NLTK is a leading platform for building Python programs to work with human language data.”
stop_words = set(stopwords.words(‘english’))
from nltk.tokenize import word_tokenize

tokens = word_tokenize(input_str)

result = [i for i in tokens if not i in stop_words]
print (result)

# Output:
# [‘NLTK’, ‘leading’, ‘platform’, ‘building’, ‘Python’, ‘programs’, ‘work’, ‘human’, ‘language’, ‘data’, ‘.’]



# 7. Remove sparse terms and particular words
# In some cases, it’s necessary to remove sparse terms or particular words from texts. 
# This task can be done using stop words removal techniques considering that 
# any group of words can be chosen as the stop words.


	# 7.1. Stemming: 
		# Stemming is a process of reducing words to their word stem, 
		# base or root form (for example, books — book, looked — look). 
		# The main two algorithms are Porter stemming algorithm (removes common morphological
		# and inflexional endings from words [14]) and Lancaster stemming 
		# algorithm (a more aggressive stemming algorithm). In the “Stemming” sheet 
		# of the table some stemmers are described.
		
		# 7.1.1. Stemming using NLTK:
		from nltk.stem import PorterStemmer
		from nltk.tokenize import word_tokenize
		stemmer= PorterStemmer()
		input_str=”There are several types of stemming algorithms.”
		input_str=word_tokenize(input_str)
		for word in input_str:
			print(stemmer.stem(word))
			
		# Output:
		# There are sever type of stem algorithm.
		
	# 7.2. Lemmatization
		# The aim of lemmatization, like stemming, is to reduce inflectional forms
		# to a common base form. As opposed to stemming, lemmatization does not simply
		# chop off inflections. Instead it uses lexical knowledge bases to get the correct base forms of words.
		
		#  Lemmatization using NLTK:
		from nltk.stem import WordNetLemmatizer
		from nltk.tokenize import word_tokenize
		lemmatizer=WordNetLemmatizer()
		input_str=”been had done languages cities mice”
		input_str=word_tokenize(input_str)
		for word in input_str:
			print(lemmatizer.lemmatize(word))
			
		# Output:
		# be have do language city mouse
		
	# 7.3. Part of speech tagging (POS)
		# Part-of-speech tagging aims to assign parts of speech to each
		# word of a given text (such as nouns, verbs, adjectives, and others) 
		# based on its definition and its context. There are many tools
		# containing POS taggers including NLTK, spaCy, TextBlob, Pattern, Stanford CoreNLP, 
		# Memory-Based Shallow Parser (MBSP), Apache OpenNLP, Apache Lucene, General Architecture 
		# for Text Engineering (GATE), FreeLing, Illinois Part of Speech Tagger, and DKPro Core.
		
		
		# Part-of-speech tagging using TextBlob:
		input_str=”Parts of speech examples: an article, to write, interesting, easily, and, of”
		from textblob import TextBlob
		result = TextBlob(input_str)
		print(result.tags)
		
		# Output:
		# [(‘Parts’, u’NNS’), (‘of’, u’IN’), (‘speech’, u’NN’), (‘examples’, u’NNS’), (‘an’, u’DT’), 
		# (‘article’, u’NN’), (‘to’, u’TO’), (‘write’, u’VB’), (‘interesting’, u’VBG’), (‘easily’, u’RB’), 
		# (‘and’, u’CC’), (‘of’, u’IN’)]
		
	# 7.4. Chunking (shallow parsing)
		# Chunking is a natural language process that identifies constituent parts of 
		# sentences (nouns, verbs, adjectives, etc.) and links them to higher order units 
		# that have discrete grammatical meanings (noun groups or phrases, verb groups, etc.) [23]. 
		# Chunking tools: NLTK, TreeTagger chunker, Apache OpenNLP, General Architecture 
		# for Text Engineering (GATE), FreeLing.
		
		# Chunking using NLTK:
			# The first step is to determine the part of speech for each word:
			input_str=”A black television and a white stove were bought for the new apartment of John.”
			from textblob import TextBlob
			result = TextBlob(input_str)
			print(result.tags)
			
			# Output:
			# [(‘A’, u’DT’), (‘black’, u’JJ’), (‘television’, u’NN’), 
			# (‘and’, u’CC’), (‘a’, u’DT’), (‘white’, u’JJ’), (‘stove’, u’NN’), 
			# (‘were’, u’VBD’), (‘bought’, u’VBN’), (‘for’, u’IN’), (‘the’, u’DT’), 
			# (‘new’, u’JJ’), (‘apartment’, u’NN’), (‘of’, u’IN’), (‘John’, u’NNP’)]
			
		# The second step is chunking:
			reg_exp = “NP: {<DT>?<JJ>*<NN>}”
			rp = nltk.RegexpParser(reg_exp)
			result = rp.parse(result.tags)
			print(result)
			
			# Output
			# (S (NP A/DT black/JJ television/NN) and/CC (NP a/DT white/JJ stove/NN) 
			# were/VBD bought/VBN for/IN (NP the/DT new/JJ apartment/NN)of/IN John/NNP)
			
		# It’s also possible to draw the sentence tree structure using code
			result.draw()
			
	# 7.5. Named entity recognition
		# Named-entity recognition (NER) aims to find named entities in text and classify
		# them into pre-defined categories (names of persons, locations, organizations, times, etc.).
		
		# Named-entity recognition using NLTK:
			from nltk import word_tokenize, pos_tag, ne_chunk
			input_str = “Bill works for Apple so he went to Boston for a conference.”
			print ne_chunk(pos_tag(word_tokenize(input_str)))
			
		# Output:
		# (S (PERSON Bill/NNP) works/VBZ for/IN Apple/NNP so/IN he/PRP went/VBD to/TO (GPE Boston/NNP) for/IN a/DT conference/NN ./.)
		
	
	# 7.6. Coreference resolution (anaphora resolution)
		# Pronouns and other referring expressions should be connected to the right individuals.
		# Coreference resolution finds the mentions in a text that refer to the same real-world entity. 
		# For example, in the sentence, “Andrew said he would buy a car” the pronoun “he” refers to the same person, 
		# namely to “Andrew”. Coreference resolution tools: Stanford CoreNLP, spaCy, Open Calais, Apache OpenNLP 
		# are described in the “Coreference resolution” sheet of the table.
		
		# https://corpling.uis.georgetown.edu/xrenner/doc/using.html#importing-as-a-module
		
	# 7.7. Collocation extraction
		# Collocations are word combinations occurring together more often than would be expected by chance. 
		# Collocation examples are “break the rules,” “free time,” “draw a conclusion,” “keep in mind,” “get ready,” and so on.
		
		# Collocation extraction using ICE
		
		input=[“he and Chazz duel with all keys on the line.”]
		from ICE import CollocationExtractor
		extractor = CollocationExtractor.with_collocation_pipeline(“T1” , bing_key = “Temp”,pos_check = False)
		print(extractor.get_collocations_of_length(input, length = 3))
		
		# Output:
		# [“on the line”]
		
	# 7.8. Relationship extraction
		# Relationship extraction allows obtaining structured information from unstructured sources 
		# such as raw text. Strictly stated, it is identifying relations (e.g., acquisition, spouse, employment) 
		# among named entities (e.g., people, organizations, locations). For example, from the 
		# sentence “Mark and Emily married yesterday,” we can extract the information that Mark is Emily’s husband.
		
		# http://www.nltk.org/howto/relextract.html
		
	
# In this post, we talked about text preprocessing and described its main steps including normalization, 
# tokenization, stemming, lemmatization, chunking, part of speech tagging, named-entity recognition, coreference resolution, 
# collocation extraction, and relationship extraction. We also discussed text preprocessing tools and examples. 
# A comparative table was created.

# After the text preprocessing is done, the result may be used for more complicated NLP tasks, 
# for example, machine translation or natural language generation.

# Resources:
		# https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908
		# http://www.nltk.org/index.html
		# http://textblob.readthedocs.io/en/dev/
		# https://spacy.io/usage/facts-figures
		# https://radimrehurek.com/gensim/index.html
		# https://opennlp.apache.org/
		# http://opennmt.net/
		# https://gate.ac.uk/
		# https://uima.apache.org/
		# https://www.clips.uantwerpen.be/pages/MBSP#tokenizer
		# https://rapidminer.com/
		# http://mallet.cs.umass.edu/
		# https://www.clips.uantwerpen.be/pages/pattern
		# https://nlp.stanford.edu/software/tokenizer.html#About
		# https://tartarus.org/martin/PorterStemmer/
		# http://www.nltk.org/api/nltk.stem.html
		# https://snowballstem.org/
		# https://pypi.python.org/pypi/PyStemmer/1.0.1
		# https://www.elastic.co/guide/en/elasticsearch/guide/current/hunspell.html
		# https://lucene.apache.org/core/
		# https://dkpro.github.io/dkpro-core/
		# http://ucrel.lancs.ac.uk/claws/
		# http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/
		# https://en.wikipedia.org/wiki/Shallow_parsing
		# https://cogcomp.org/page/software_view/Chunker
		# https://github.com/dstl/baleen
		# https://github.com/CogComp/cogcomp-nlp/tree/master/ner
		# https://github.com/lasigeBioTM/MER
		# https://blog.paralleldots.com/product/dig-relevant-text-elements-entity-extraction-api/
		# http://www.opencalais.com/about-open-calais/
		# http://alias-i.com/lingpipe/index.html
		# https://github.com/glample/tagger
		# http://minorthird.sourceforge.net/old/doc/
		# https://www.ibm.com/support/knowledgecenter/en/SS8NLW_10.0.0/com.ibm.watson.wex.aac.doc/aac-tasystemt.html
		# https://www.poolparty.biz/
		# https://www.basistech.com/text-analytics/rosette/entity-extractor/
		# http://www.bart-coref.org/index.html
		# https://wing.comp.nus.edu.sg/~qiu/NLPTools/JavaRAP.html
		# http://cswww.essex.ac.uk/Research/nle/GuiTAR/
		# https://www.cs.utah.edu/nlp/reconcile/
		# https://github.com/brendano/arkref
		# https://cogcomp.org/page/software_view/Coref
		# https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30
		# https://github.com/smartschat/cort
		# http://www.hlt.utdallas.edu/~altaf/cherrypicker/
		# http://nlp.lsi.upc.edu/freeling/
		# https://corpling.uis.georgetown.edu/xrenner/#
		# http://takelab.fer.hr/termex_s/
		# https://www.athel.com/colloc.html
		# http://linghub.lider-project.eu/metashare/a89c02f4663d11e28a985ef2e4e6c59e76428bf02e394229a70428f25a839f75
		# http://ws.racai.ro:9191/narratives/batch2/Colloc.pdf
		# http://www.aclweb.org/anthology/E17-3027
		# https://metacpan.org/pod/Text::NSP
		# https://github.com/knowitall/reverb
		# https://github.com/U-Alberta/exemplar
		# https://github.com/aoldoni/tetre
		# https://www.textrazor.com/technology
		# https://github.com/machinalis/iepy
		# https://www.ibm.com/watson/developercloud/natural-language-understanding/api/v1/#relations
		# https://github.com/mit-nlp/MITIE