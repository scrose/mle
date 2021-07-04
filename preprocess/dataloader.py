import json
import pandas as pd
from pandas.io.json import json_normalize
from tqdm import tqdm
import string

''' Preprocess SQuAD input datasets '''
    
def preprocess_squad(input_file):
    #load SQuAD dataset
    with open(input_file) as f:
        d = json.load(f)
    f.close()

    # normalize data (document) chunks
    df = json_normalize(d, record_path='data')
    
    # normalize paragraphs
    paragraphs = []
    for i, row in enumerate(df['paragraphs']):
        # convert json data to dataframe
        df_row = json_normalize(json.loads(json.dumps(row)))
        paragraphs += [df_row]
    
    df = pd.concat(paragraphs)
    
    src_array = []
    #sen_array = []
    tgt_array = []
    ans_array = []
    
    # extract paragraphs
    for i, row in enumerate(tqdm(df.itertuples(), desc='Progress: ')):
        # parse context
        context = row.context.split('.').copy()
        #ctxt = row.context

        # convert json data to dataframe
        df_qas = json_normalize(json.loads(json.dumps(row.qas)))
        
        # extract questions and map to context
        for j, qas in enumerate(df_qas.itertuples()):
            df_ans = json_normalize(json.loads(json.dumps(qas.answers)))
            answer = df_ans.iloc[0]['text']
            
            # filter unanswerable questions
            if len(answer) > 0:
                # get index of answer keyword
                index = df_ans.iloc[0]['answer_start']
                # include sentence
                src_array += [extract_sentence(index, context)]
                # add question to target array
                tgt_array += [rmv_punc(qas.question)]
                # create answer mask
                #context, answer = ans_mask(ctxt[:], index, answer)
                ans_array += [rmv_punc(answer)]
                # include full context
                #src_array += [rmv_punc(ctxt)]
        
    df_src = pd.DataFrame({'context': src_array})
    #df_sen = pd.DataFrame({'sentence': sen_array})
    df_tgt = pd.DataFrame({'question': tgt_array})
    df_ans = pd.DataFrame({'answer': ans_array})
    
    # reassign indexes
    df_src.index = range(len(df_src.index))
    #df_sen.index = range(len(df_src.index))
    df_tgt.index = range(len(df_tgt.index))
    df_ans.index = range(len(df_ans.index))

    # combine source and target
    df_combined = pd.concat([df_src, df_tgt, df_ans], axis=1)

    return df_combined

def preprocess(config):
    """
    Preprocess input SQuAD files
    """

    print('Preprocessing SQuAD data files...')

    # process data
    df_train = preprocess_squad(config.train_raw)
    df_train.to_csv(config.train_data, index=False)

    
    df_valid = preprocess_squad(config.valid_raw)
    df_valid.to_csv(config.valid_data, index=False)

    # Randomly sample 1000 rows from validation dataset
    df_test = preprocess_squad(config.test_raw)
    df_test = df_test.sample(n=100)
    df_test.to_csv(config.test_data, index=False)
    
    print('training data processed / dim: {}'.format(df_train.shape))
    print('validation data processed / dim: {}'.format(df_valid.shape))
    print('test data processed / dim: {}'.format(df_test.shape))


# Helper Functions

def extract_sentence(index, context):
    char_count = 0
    for s in context:
        char_count += len(s)
        if index < char_count:
            return s.lower()
        
# pad punctuation
def rmv_punc(s):
    return s.translate(str.maketrans('', '', string.punctuation))
