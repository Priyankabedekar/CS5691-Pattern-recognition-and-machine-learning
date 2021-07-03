import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import Spam_Classifier_Training
from os import path
import csv


lemmatise = CountVectorizer(stop_words=["all", "in", "the", "is", "and"])


def wordsinTestMail(dict1, file):
    """
    Description: Calculate occurrences of all dictionary words in input test mail
    Parameters: Dictionary; Input test mail
    Return: Result list - Corresponding index of word in resList matches no. of occurrences in input test mail
    """
    resList = [0] * len(dict1)
    file = lemmatise.fit_transform(file)
    count = file.toarray()
    words = lemmatise.get_feature_names()
    for i, word in enumerate(words):
        if word in dict1:
            resList[dict1.index(word)] += 1
    return resList


def classify(dict1, pSpamWords, pHamWords, pSpam, test_target):
    """
    Description: Classifies labels of input test mail
    Parameters: Dictionary; Spam/Ham probability of dictionary words, Spam probability of mails, test mail
    Return: 1 or 0 (Spam or Ham)
    """
    words_count = wordsinTestMail(dict1, test_target)
    target_words = np.array(words_count)
    p1 = sum(target_words * pSpamWords) + np.log(pSpam)
    p0 = sum(target_words * pHamWords) + np.log(1 - pSpam)
    if p1 > p0:
        return 1
    else:
        return 0


def predictTest():
    if path.exists("Dictionary.txt"):
        file = open("Dictionary.txt", "r")
        lemmatise.fit_transform(file)
        dict1 = lemmatise.get_feature_names()
        print(dict1)
    else:
        dict1 = Spam_Classifier_Training.createDict()
    p_Spamwords = np.zeros(len(dict1)) 
    p_Hamwords = np.zeros(len(dict1))
    p_Spam = 0
    if path.exists("P_Spam.txt"):
        with open('P_Spam.txt', 'r') as f2:
            p_Spam = float(f2.read())
    print(p_Spam)
    if path.exists("P_Spam_Words"):
        p_Spamwords = np.loadtxt("./P_Spam_Words")
    if path.exists("P_Spam_Words"):
        p_Hamwords = np.loadtxt("./P_Ham_Words")
    if p_Hamwords.all() == 0 or p_Spamwords.all == 0 or p_Spam == 0:
        words_matrix = Spam_Classifier_Training.wordsinAllMails(dict1)
        p_Spamwords, p_Hamwords, p_Spam = Spam_Classifier_Training.trainNB(words_matrix)
    else:
        print('Showtime.')
    test_dir=os.getcwd()+'/Test'
    test_label=np.zeros(len(os.listdir(test_dir)))
    test_emails = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
    for i, file in enumerate(test_emails):
        with open(file) as m:
            test_label[i] = classify(dict1, p_Spamwords, p_Hamwords, p_Spam, m)

    test_label_true = [0] * len(os.listdir(test_dir))
    for i, file in enumerate(os.listdir(test_dir)):
        if file.startswith('spm'):
            test_label_true[i] = 1
    print('Correct:',580 - sum(abs(test_label_true - test_label)))
    print('Incorrect:',sum(abs(test_label_true - test_label)))
    print('Accuracy :',100 - (sum(abs(test_label_true - test_label)) / len(os.listdir(test_dir)) * 100), '%')

    result = np.savetxt('output.txt', test_label, fmt='%.0f')
    os.system("notepad output.txt")
    final = open('output.txt', 'r')
    # print(final.read())


predictTest()