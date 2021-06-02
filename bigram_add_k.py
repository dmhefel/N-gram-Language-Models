# CS114 Spring 2020 Programming Assignment 3
# N-gram Language Models

from collections import defaultdict
from languageModel import LanguageModel
import numpy as np
from scipy.sparse import lil_matrix

class BigramAddK(LanguageModel):

    def __init__(self):
        self.word_dict = {}
        self.total = {}
        self.prob_counter = None
        self.k = .75

    '''
    Trains a bigram-add-k language model on a training set.
    '''
    def train(self, trainingSentences):
        word_counts = defaultdict(lambda: defaultdict(int))
        listofwords = []

        for sentence in trainingSentences:
            previous_word = LanguageModel.STOP
            for word in sentence:
                word_counts[previous_word][word] += 1
                listofwords.append(word)
                previous_word = word
            word_counts[previous_word][LanguageModel.STOP] += 1
            
        listofwords.append(LanguageModel.STOP)
        listofwords.append(LanguageModel.UNK)



        for i, word in enumerate(sorted(list(set(listofwords)))):
            self.word_dict[word] = i
        self.prob_counter = lil_matrix((len(self.word_dict), len(self.word_dict)))

        for k, v in word_counts.items():
            for k1,v1 in v.items():
                self.prob_counter[self.word_dict[k], self.word_dict[k1]] = word_counts[k][k1]
        for i in range(len(self.word_dict)):
            self.prob_counter[self.word_dict[LanguageModel.UNK], i] = 1
            self.prob_counter[i, self.word_dict[LanguageModel.UNK]] = 1
        
        #self.total[previous_word] = self.prob_counter[previous_word].sum()
        for i in range(len(self.word_dict)):
            self.total[i] = self.prob_counter[i].sum()
            self.total[i] += (self.k*len(self.word_dict))
            self.prob_counter[i] = self.prob_counter[i].multiply(1 / self.total[i]).tolil()
            

    '''
    Returns the probability of the word at index, according to the model, within
    the specified sentence.
    '''
    def getWordProbability(self, sentence, index):
        if index == 0:
            previous_word = LanguageModel.STOP
        else:
            previous_word = sentence[index-1]
        if index == len(sentence):
            word = LanguageModel.STOP
        else:
            word = sentence[index]
        if word not in self.word_dict:
            word = LanguageModel.UNK
        if previous_word not in self.word_dict:
            previous_word = LanguageModel.UNK
        return self.prob_counter[self.word_dict[previous_word], self.word_dict[word]]+(self.k/self.total[self.word_dict[previous_word]])

    '''
    Returns, for a given context, a random word, according to the probabilities
    in the model.
    '''
    def generateWord(self, context):
        if len(context)<1:
            index = self.word_dict[LanguageModel.STOP]
        else:
            index = self.word_dict[context[-1]]
        probs = self.prob_counter[index].toarray().ravel()
        probs = probs +(self.k/(self.total[index]))
        
        return np.random.choice(sorted(self.word_dict), p=probs)
