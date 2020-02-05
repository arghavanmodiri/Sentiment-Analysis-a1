import sys
import argparse
import os
import json
import re
import spacy
import html
import pandas as pd


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment , steps=range(1, 5)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment
    if 1 in steps:  # replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)
    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)
    if 4 in steps:  # remove duplicate spaces
        modComm = re.sub(' +', ' ', modComm)

    # TODO: get Spacy document for modComm
    modComm = nlp(modComm)

    output = []
    for sent in modComm.sents:
        print("senteence : ", sent)
        for token in sent:
            if token.lemma_[0] == "-":
                output.append(token.text)
            else:
                output.append(token.lemma_)
            output.append("/{}".format(token.tag_))
            output.append(" ")
        output = output[:-1]
        output.append("\n")

    modComm = "".join(output)
    print(modComm)
    print("\n")

    return modComm


def main(args):
    allOutput = []
    data_df = pd.DataFrame()
    print(os.walk(indir))
    for subdir, dirs, files in os.walk(indir):
        print("dfs")
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))
            print('1 ', len(data))
            start = args.ID[0] % len(data)
            #data = data[start:start+args.max]
            #data = data[start:start+4]
            print('2 ', len(data))
            data_temp = []
            for line in data:
                data_temp.append(json.loads(line))
            data_df = pd.DataFrame.from_dict(data_temp, orient='columns')
            data_df = data_df[['id','body']]
            data_df['cat'] = file
            # print(data_df.iloc[0:3]['body'])
            print(data_df['body'])
            data_df['body'] = data_df['body'].apply(lambda x:preproc1(x))
            #print(data_df.iloc[0:3]['body'])
            allOutput = data_df.to_dict(orient='records')

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    main(args)
