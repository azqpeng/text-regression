
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import numpy as np
from collections import OrderedDict
from itertools import combinations

def processDocs(datapath):
    """
    This function takes the path to text file and returns a list of documents, and a list of document names
    """
    docList = [] # get_vocab(datapath + "_docs.txt")
    f = open(datapath, 'rb')
    for line in f.readlines():
        docList.append(line.decode('UTF-8').split('\t')[1].strip())
        # docNames.append(line.decode('UTF-8').split('\t')[0].strip())
    f.close()

    return docList

    


def build_graph(docList, wordEdges = "PMI", window = 10):
    """
    This function takes the dataset and generates an adjacency matrix based on the specifications in Yao et al. (2019).
    Input: string representing path to the dataset, not including entire filename and only it's prefix (e.g. "guten")
    """

    # initialize variables
    numDocs = len(docList)
    wordList, tfvect = get_vocab(docList)
    numWords = len(wordList)
    numNodes =  numWords + numDocs
    print("This is the: " + str(numWords))

    # build empty adjacency matrix (sparse) # note here that the first numWords indices are words, and the last numDocs indices are documents
    adj = np.identity(numNodes)
    
    # TO DO: convert above into sparse matrix.

    # build word-to-doc edges
    if wordEdges == "PMI":
        adj = PMIEdges(docList, wordList, window, adj)
    if wordEdges == "w2v":
        adj = w2vEdges(docList, wordList, window, adj)
    if wordEdges == "BERT":
        adj = BERTEdges(docList, wordList, window, adj)

    # build word-to-doc edges using TF-IDF
    tfiter = tfvect.toarray()
    for words in range(tfiter.shape[1]):
        for docs in range(tfiter.shape[0]):
            if tfiter[docs, words] > 0:
                adj[words, docs + numWords] = tfiter[docs, words]
                adj[docs+numWords, words] = tfiter[docs, words]
    print("TFIDF is complete")

    # return adjacency matrix using A^ = D^-1/2 * A * D^-1/2
    diag = np.diag(np.power(np.sum(adj, axis = 1), -0.5))
    adj = np.matmul(np.matmul(diag, adj), diag)

    # adj = sparse.csr_matrix(adj)

    return adj

def PMIEdges(docList, wordList, window, adj):

    wordList = tuple(wordList)
    wordSet = set(wordList)

    # initializations
    n_i  = OrderedDict((name, 0) for name in wordList)
    word2index = OrderedDict((name,index) for index,name in enumerate(wordList))
    occurrences = np.zeros( (len(wordList),len(wordList)) ,dtype=np.int32)

    counter = 0

    # count word occurences and co-occurences
    for l in docList:
        docSplit = l.split()
        for i in range(len(docSplit) - window + 1):
            # total windows #W
            counter +=1
            d = docSplit[i:i+window]
            e = set()
            # occurences of words #W(i))
            for word in d:
                if word in wordSet:
                    n_i[word] += 1
                    e.add(word)
            # co-occurences of words #W(i, j)
            for w1,w2 in combinations(e,2):
                i1 = word2index[w1]
                i2 = word2index[w2]
                occurrences[i1][i2] += 1
                occurrences[i2][i1] += 1

    
    pmi = occurrences/counter

    for word in n_i:
        if n_i[word] == 0:
            print(word)

    # perform the computations
    p_i = np.array(list(n_i.values()))/counter
    for col in range(len(wordList)):
        pmi[:, col] = pmi[:, col]/p_i[col]
    for row in range(len(wordList)):
        pmi[row, :] = pmi[row,:]/p_i[row]
    pmi = pmi + 1e-9
    for col in range(len(wordList)):
        pmi[:, col] = np.log(pmi[:, col])

    print("computations complete")
    

    # add into adjacency matrix
    for i in range(len(wordList)):
        for j in range(len(wordList)):
            if i == j:
                adj[i,j] = 1
            elif pmi[i, j] > 0:
                adj[i,j] = pmi[i,j]

    print("PMI is complete")
    return adj

def w2vEdges(docList, wordList, adj):
    pass

def BERTEdges(docList, wordList, adj):
    pass



def get_vocab(docList):
    """
    This function takes the dataset and generates the list of vocab with their number of appearances as well using CountVectorizer.
    """

    # docList = [] # get_vocab(datapath + "_docs.txt")
    # docNames = []
    # f = open(docPath, 'rb')
    # for line in f.readlines():
    #     docList.append(line.decode('UTF-8').split('\t')[1].strip())
    #     docNames.append(line.decode('UTF-8').split('\t')[0].strip())
    # f.close()


    tfidf = TfidfVectorizer(min_df=3)
    tfidfvect = tfidf.fit_transform(docList)
    vocab = tfidf.get_feature_names_out()
    # df_tfidfvect = pd.DataFrame(data = tfidfvect.toarray(),index = docNames, columns = vocab)
    print("Vocab is complete")

    return vocab, tfidfvect # df_tfidfvect


# def get_vocab(textpath):
#     """
#     This function takes the dataset and generates the list of vocab with their number of appearances as well.
#     """
#     vocabfreq1 = defaultdict(int)
#     vocabfreq2 = defaultdict(int)
#     vocabtexts1 = defaultdict(set)
#     vocabtexts2 = defaultdict(set)
#     with open(textpath, 'r') as corpora:
#         for doc in corpora:
#             doc = doc.strip().split() 
#             for word in doc:
#                 vocabfreq1[word] += 1
#                 vocabtexts1[word].add(doc)
    
#     for word in vocabfreq1:
#         if vocabfreq1[word] >= 5:
#             if len(vocabtexts1[word]) > 1:
#                 vocabfreq2[word] = vocabfreq1[word]
#                 vocabtexts2[word] = vocabtexts1[word]
#     print(len(vocabfreq2))
#     return vocabfreq2, vocabtexts2


# build_graph('data/gutenold/guten_sentences.txt', "PMI", 5)


