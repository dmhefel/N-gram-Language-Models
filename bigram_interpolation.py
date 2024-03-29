# CS114 Spring 2020 Programming Assignment 3
# N-gram Language Models

from collections import defaultdict
from languageModel import LanguageModel
import numpy as np
from unigram import Unigram
from bigram import Bigram

class BigramInterpolation(LanguageModel):

    def __init__(self):
        self.unigram = Unigram()
        self.bigram = Bigram()
        # just needed for languageModel.py to work
        self.word_dict = self.bigram.word_dict
        self.lambda_1 = 0.6
        self.lambda_2 = 0.4

    '''
    Trains a bigram-interpolation language model on a training set.
    '''
    def train(self, trainingSentences):
        self.unigram.train(trainingSentences)
        self.bigram.train(trainingSentences)

    '''
    Returns the probability of the word at index, according to the model, within
    the specified sentence.
    '''
    def getWordProbability(self, sentence, index):
        return (self.lambda_1*self.bigram.getWordProbability(sentence, index)
                +self.lambda_2*self.unigram.getWordProbability(sentence, index))

    '''
    Returns, for a given context, a random word, according to the probabilities
    in the model.
    '''
    def generateWord(self, context):
        probs_uni = self.unigram.prob_counter[0].toarray().ravel()
        probs_uni *= self.lambda_2
        if len(context)<1:
            index = self.bigram.word_dict[LanguageModel.STOP]
        else:
            index = self.bigram.word_dict[context[-1]]
        probs_bi = self.bigram.prob_counter[index].toarray().ravel()
        probs_bi *= self.lambda_1
        probs = np.add(probs_bi, probs_uni)

        
        return np.random.choice(sorted(self.word_dict), p=probs)
