import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


lemmatise = CountVectorizer(stop_words=["all", "in", "the", "is", "and"])


def trainDir():
    """
    Description: Returns training directory
    Parameter: Nil
    """
    train_dir = os.getcwd() + '/Final/Train_Temp'
    return train_dir

def createDict():
    """
    Description: Creates dictionary of words based on all training mails
    Parameter: Nil
    Return: Dictionary of words as list
    """
    train_dir = trainDir()
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words
    lemmatise.fit_transform(all_words)
    dictionary = lemmatise.get_feature_names()
    dictionary = [item for item in dictionary if item.isalpha()]
    with open("Dictionary.txt", "w") as output:
        output.write(str(dictionary))
    return dictionary


def trainLabel():
    train_dir = trainDir()
    email_label = [0] * len(os.listdir(train_dir))
    for i, file in enumerate(os.listdir(train_dir)):
        if file.startswith('spm'):
            email_label[i] = 1
    return email_label


def wordsinOneMail(dict1, inputMail):
    """
    Description: Calculate occurrences of all dictionary words in input training mail
    Parameter: Dictionary; Input training mail
    Return: Result list - Corresponding index of word in resList matches no. of occurrences in input training mail
    """
    resList = [0] * len(dict1)
    for i, line in enumerate(inputMail):
        if i == 2:
            words = line.split()
            for word in words:
                if word in dict1:
                    resList[dict1.index(word)] += 1
    return resList


def wordsinAllMails(dict1):
    """
    Description: Create a matrix of occurrences of dictionary words of all training mails
    Parameters: Dictionary; Training mails directory
    Return: A matrix of occurrences of dictionary words corresponding to all training mails
    """
    train_dir = trainDir()
    wordsinMails = []
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    for file in emails:
        with open(file) as m:
            setOfwords = wordsinOneMail(dict1, m)
            wordsinMails.append(setOfwords)
    return np.array(wordsinMails)


def trainNB(words_matrix):
    """
    Description: Calculate probability of spam mails, probability of all dictionary words in spam and ham mails
    Parameters: Matrix of occurrences of dictionary words corresponding to all training mails; Training mail labels
    Return: Probabilities of spam mails, all words in spam mails, all words in ham mails
    """
    email_label = trainLabel()
    mails_count = len(words_matrix)
    words_count = len(words_matrix[0])
    pSpam = sum(email_label) / float(mails_count)
    dict_spamcount = np.ones(words_count)
    dict_hamcount = np.ones(words_count)
    spamwords_count = 1.0
    hamwords_count = 1.0
    for i in range(0, mails_count):
        if email_label[i] == 1:
            dict_spamcount += words_matrix[i]
            spamwords_count += sum(words_matrix[i])
        else:
            dict_hamcount += words_matrix[i]
            hamwords_count += sum(words_matrix[i])
    pSpamWords = np.log(dict_spamcount / spamwords_count)
    pHamWords = np.log(dict_hamcount / hamwords_count)
    np.savetxt("P_Spam_Words", pSpamWords)
    np.savetxt("P_Ham_Words", pHamWords)
    with open('P_Spam.txt', 'w') as f:
        f.write('%f' % pSpam)
    return pSpamWords, pHamWords, pSpam