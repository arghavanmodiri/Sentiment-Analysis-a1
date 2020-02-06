import numpy as np
import argparse
import json
import re
import pandas as pd
import string

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

bristol_df = pd.read_csv("/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv", index_col='WORD')
warringer_df = pd.read_csv("/u/cs401/Wordlists/Ratings_Warriner_et_al.csv", index_col='Word')

with open('/u/cs401/A1/feats/Alt_IDs.txt', 'r') as f:
    alt_ids = f.read().splitlines()
with open('/u/cs401/A1/feats/Left_IDs.txt', 'r') as f:
    left_ids = f.read().splitlines()
with open('/u/cs401/A1/feats/Right_IDs.txt', 'r') as f:
    right_ids = f.read().splitlines()
with open('/u/cs401/A1/feats/Center_IDs.txt', 'r') as f:
    center_ids = f.read().splitlines()

alt_feats = np.load('/u/cs401/A1/feats/Alt_feats.dat.npy')
right_feats = np.load('/u/cs401/A1/feats/Right_feats.dat.npy')
left_feats = np.load('/u/cs401/A1/feats/Left_feats.dat.npy')
center_feats = np.load('/u/cs401/A1/feats/Center_feats.dat.npy')


def extract1(comment, id, cat):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''    
    # TODO: Extract features that rely on capitalization.
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # TODO: Extract features that do not rely on capitalization.
    feats = np.zeros(173+1)
    if comment == '':
        return feats[1:]

    modComm = comment
    modComm = re.sub(r"/[-a-zA-Z!\"#$%&'`()*+,.:;<=>?@\[\]^_~]+?\s", " ", modComm)
    modComm = re.sub(r"/[-a-zA-Z!\"#$%&'`()*+,.:;<=>?@\[\]^_~]+?\n", " ", modComm)
    words = modComm.split()

    modComm = " " + comment
    modComm = re.sub(r"\s.+?/", " ", modComm)
    modComm = re.sub(r"\n.+?/", " ", modComm)
    modComm = modComm[1:]
    tags = modComm.split()

    token_length_sum = 0
    token_count = 0
    aoa = []
    img = []
    fam = []
    valence = []
    arousal = []
    dominance = []

    if len(tags) != len(words):
        print("*******************")
        print(id)
        print(cat)
        print("comment :", repr(comment))

    for idx in range(len(words)):
        word = words[idx]
        tag = tags[idx]
        if word.isupper() and len(word) >= 3 and word.isalpha():
            feats[1] += 1
        word = word.lower()
        words[idx] += word
        if word in FIRST_PERSON_PRONOUNS:
            feats[2] += 1
        if word in SECOND_PERSON_PRONOUNS:
            feats[3] += 1
        if word in THIRD_PERSON_PRONOUNS:
            feats[4] += 1
        if tag == 'CC':
            feats[5] += 1
        if tag == 'VBD':
            feats[6] += 1
        if word in {"'ll", "will", "gonna"}:
            feats[7] += 1
        if word == "going" and idx < len(words)-2:
            if words[idx+1] == "to" and tag[idx+2] == "VB":
                feats[7] += 1
        feats[8] += word.count(',')
        if len(word)>1 and len([ch for ch in word if all(j in string.punctuation for j in ch)]) == len(word):
            feats[9] += 1
        if tag in {'NN', 'NNS'}:
            feats[10] += 1
        if tag in {'NNP', 'NNPS'}:
            feats[11] += 1
        if tag in {'RB', 'RBR', 'RBS'}:
            feats[12] += 1
        if tag in {'WDT', 'WP', 'WP$', 'WRB'}:
            feats[13] += 1
        if word in SLANG:
            feats[14] += 1
        if len([ch for ch in word if all(j in string.punctuation for j in ch)]) != len(word):
            token_length_sum += len(word)
            token_count += 1
        if word in bristol_df.index:
            aoa.append(bristol_df.loc[word]['AoA (100-700)'].max())
            img.append(bristol_df.loc[word]['IMG'].max())
            fam.append(bristol_df.loc[word]['FAM'].max())
        if word in warringer_df.index:
            valence.append(warringer_df.loc[word]['V.Mean.Sum'].max())
            arousal.append(warringer_df.loc[word]['A.Mean.Sum'].max())
            dominance.append(warringer_df.loc[word]['D.Mean.Sum'].max())

    modComm = comment
    #modComm = re.sub(r"\s.+?/", " ", modComm)
    #modComm = re.sub(r"\n.+?/", "\n", modComm)
    #modComm = modComm[1:]
    sents = modComm.split("\n")
    sents = sents[:-1]
    sents_sum = 0
    for sent in sents:
        sents_sum += len(sent.split())

    if token_count > 0:
        feats[16] = 1.0*token_length_sum / token_count

    feats[15] = 1.0*sents_sum / len(sents)
    feats[17] = len(sents)
    if aoa != []:
        feats[18] = np.mean(np.array(aoa))
        feats[21] = np.std(np.array(aoa))
    if img != []:
        feats[19] = np.mean(np.array(img))
        feats[22] = np.std(np.array(img))
    if fam != []:
        feats[20] = np.mean(np.array(fam))
        feats[23] = np.std(np.array(fam))
    if valence != []:
        feats[24] = np.mean(np.array(valence))
        feats[27] = np.std(np.array(valence))
    if arousal != []:
        feats[25] = np.mean(np.array(arousal))
        feats[28] = np.std(np.array(arousal))
    if dominance != []:
        feats[26] = np.mean(np.array(dominance))
        feats[29] = np.std(np.array(dominance))
    if token_count == 0:
        print("comment :", comment)
        print(feats[1:])

    return feats[1:]

def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''

    if comment_class == 'Alt':
        row = alt_ids.index(comment_id)
        feats_liwc = alt_feats[row]
    elif comment_class == 'Right':
        row = right_ids.index(comment_id)
        feats_liwc = right_feats[row]
    elif comment_class == 'Left':
        row = left_ids.index(comment_id)
        feats_liwc = left_feats[row]
    elif comment_class == 'Center':
        row = center_ids.index(comment_id)
        feats_liwc = center_feats[row]
    else:
        print("WRONG CLASS!")

    feats[29:] = feats_liwc
    return feats

def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    # TODO: Use extract1 to find the first 29 features for each 
    # data point. Add these to feats.
    # TODO: Use extract2 to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    idx = 0
    cat_dict = {'Left':0, 'Center': 1, 'Right': 2, 'Alt':3}
    for line in data:
        new_feat = extract1(line['body'], line['id'], line['cat'])
        feats[idx][:-1] = extract2(new_feat, line['cat'], line['id'])
        feats[idx][-1] = cat_dict[line['cat']]
    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

