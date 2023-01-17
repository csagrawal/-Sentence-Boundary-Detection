import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
import collections
import sys


def load_data(filename):    
    file = open(filename, "r")
    lines = file.readlines()
    index = 0 
    X_train = []
    Y_train = []
    indexes_train = []
    for line in lines:
        line_info = line.split()
        """
        True = EOS
        False = NEOS
        """
        next_word = lines[index+1].split()[1] if index < len(lines)-1 else " "
        index += 1
        if line_info[1][-1] == ".":
            next_entry = {
                "left_word": line_info[1][:-1],
                "right_word": next_word,
                "left_word_len": True if  len(line_info[1][:-1]) > 3 else False,
                "left_capitalized": line_info[1][0].isupper(),
                "right_capitalized": (not next_word[0].isupper()) if next_word else True,
                "right_word_period": True if(next_word[-1]=="." and "." not in line_info[1][:-1]) else False,
                "left_word_eos_frequency": 0.00,
                "left_word_has_period": True if("." in line_info[1][:-1] and ((not next_word[0].isupper()) if next_word else True) ) else False
            }
            Y_train.append({"result":  line_info[2]})
            X_train.append(next_entry)
            indexes_train.append({"index": line_info[0],
                                 "left_word": line_info[1][:-1],
                                 "right_word": next_word})
    COLUMN_NAMES=["left_word", "right_word", "left_word_len", "left_capitalized", "right_capitalized","right_word_period","left_word_eos_frequency","left_word_has_period"]
    x_train = pd.DataFrame(X_train, columns=COLUMN_NAMES)
    y_train = pd.DataFrame(Y_train, columns=["result"])
    return x_train, y_train , indexes_train

def generate_corpus(x_train, y_train):
    left_word_EOS_corpus = set()
    left_word_NEOS_corpus = set()
    right_word_EOS_corpus = set()
    right_word_NEOS_corpus = set()
    counts = collections.Counter(x_train[:]['left_word'])
    total_ct = len(x_train[:]['left_word'])
    for index, row in x_train.iterrows(): 
        if (y_train.iloc[index].result == 'EOS'):
            left_word_EOS_corpus.add(row[0])
            right_word_EOS_corpus.add(row[1])
        else:
            left_word_NEOS_corpus.add(row[0])
            right_word_NEOS_corpus.add(row[1])
    count = 0
        
    left_word_Common_corpus = left_word_EOS_corpus.intersection(left_word_NEOS_corpus)  
    right_word_Common_corpus = right_word_EOS_corpus.intersection(right_word_NEOS_corpus)  
    
    return left_word_EOS_corpus,left_word_NEOS_corpus,right_word_EOS_corpus,right_word_NEOS_corpus, left_word_Common_corpus,right_word_Common_corpus, counts, total_ct


def encode_data(left_word_EOS_corpus,left_word_NEOS_corpus,right_word_EOS_corpus,right_word_NEOS_corpus, left_word_Common_corpus,right_word_Common_corpus,counts, total_ct, x_train, y_train):
    """
    0 = NEOS
    1 = EOS
    2 = Common
    3 = None
    """
    for index, row in x_train.iterrows():
        x_train.loc[index,'left_word_eos_frequency'] = (counts[row[0]]/total_ct) if (row[0] in counts and row[0] in left_word_Common_corpus) else 0.0
        
        if(row[0] in left_word_Common_corpus):
            x_train.loc[index,'left_word']=2
        elif(row[0] in left_word_EOS_corpus):
            x_train.loc[index,'left_word']=1
        elif(row[0] in left_word_NEOS_corpus):
            x_train.loc[index,'left_word']=0
        else:
            x_train.loc[index,'left_word']=3

        if(row[1] in right_word_Common_corpus):
            x_train.loc[index,'right_word']=2
        elif(row[1] in right_word_EOS_corpus):
            x_train.loc[index,'right_word']=1
        elif(row[1] in right_word_NEOS_corpus):
            x_train.loc[index,'right_word']=0
        else:
            x_train.loc[index,'right_word']=3

        if(y_train.iloc[index].result == 'EOS'):
            y_train.loc[index,"result"] =True
        else:
            y_train.loc[index,"result"] = False

    return x_train, y_train

def generate_output(test_file_name, indexes_test, predictions):
    out_file = open("SBD.test.out", "a")
    test_file =  open(test_file_name, "r")
    lines = test_file.readlines()
    index = 0 
    for line in lines:
        line_info = line.split()
        if(line_info[0] == indexes_test[index]['index']):
            next_line =line_info[0]+ " " + line_info[1] + ("  EOS" if predictions[index] else "  NEOS") + "\n"
            if index <(len(predictions)-1):
                index += 1
        else:
            next_line = line_info[0]+ " " + line_info[1] +"\n"

        out_file.write(next_line)

    out_file.close()
    test_file.close()


if __name__ == "__main__":
    
    arguments = sys.argv  
    train_file = str(arguments[1])
    test_file = str(arguments[2])
    #load data
    x_train, y_train, indexes_train = load_data(train_file)
    x_test, y_test, indexes_test = load_data(test_file)


    #generate word buckets
    left_word_EOS_corpus,left_word_NEOS_corpus,right_word_EOS_corpus,right_word_NEOS_corpus, left_word_Common_corpus,right_word_Common_corpus, counts, total_ct = generate_corpus(x_train, y_train)

    #encode data 
    x_train, y_train = encode_data(left_word_EOS_corpus,left_word_NEOS_corpus,right_word_EOS_corpus,right_word_NEOS_corpus, left_word_Common_corpus,right_word_Common_corpus,counts, total_ct, x_train, y_train)
    x_test, y_test = encode_data(left_word_EOS_corpus,left_word_NEOS_corpus,right_word_EOS_corpus,right_word_NEOS_corpus, left_word_Common_corpus,right_word_Common_corpus,counts, total_ct, x_test, y_test)

    #decision tree
    dtree = DecisionTreeClassifier()
    dtree.fit(x_train,y_train)
    predictions = dtree.predict(x_test)

    print("accuracy = " + str(accuracy_score(y_test,predictions)*100))

    
    generate_output(test_file, indexes_test, predictions)

    
