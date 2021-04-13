import gensim
import allennlp_models.tagging
from allennlp.predictors.predictor import Predictor
from flask import Flask
from flask import request
import traceback
import json
from os import path
import pandas as pd
import numpy as np

import alignutil

# NLP model imports...
print('=== STARTING NLP MODEL IMPORTS ===')

# # TODO-REFERENCE originally from analyze.ipynb
# # For sentence tokenization
# from nltk import tokenize

# # TODO-REFERENCE originally from analyze.ipynb
# # For coreference resolution
# from allennlp.predictors.predictor import Predictor
# import allennlp_models.coref
# coref_predictor = Predictor.from_path(
#     "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
# )

# TODO-REFERENCE originally from analyze.ipynb
# For constituency parsing
constituency_predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
)

# # TODO-REFERENCE originally from analyze.ipynb
# from allennlp.predictors.predictor import Predictor
# import allennlp_models.structured_prediction
# # For dependency parsing
# dependency_predictor = Predictor.from_path(
#     "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
# )

# TODO-REFERENCE originally from alignment.ipynb
# Load fasttext-wiki-news-subwords-300 pretrained model
fasttext = gensim.models.keyedvectors.FastTextKeyedVectors.load(
    'model/fasttext-wiki-news-subwords-300.model', mmap='r'
)

# # TODO-REFERENCE originally from alignment.ipynb
import spacy
sp = spacy.load('en_core_web_sm')
import scispacy
from scispacy.linking import EntityLinker
scisp = spacy.load('en_core_sci_sm')
linker = scisp.add_pipe('scispacy_linker', config={'resolve_abbreviations': True, 'linker_name': 'umls'})

print('=== FINISHED NLP MODEL IMPORTS ===')

# Flask-specific code...
app = Flask(__name__)


@app.route('/api/textalign', methods=['GET'])
def api_textalign():
    print('... called /api/textalign ...')
    # retrieve arguments
    try:
        # arg_id = int(request.args['id'])
        # arg_input = request.args['input'].split('\n') if ('input' in request.args) else ['default']
        arg_input = request.args['input'].split('\n')
    except:
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback': f'{traceback.format_exc()}'
        }
    # TODO this handles a single line of input poorly
    output = {}
    # retrieve the constituency parse information
    output['parse_constituency'] = dict(zip(
        range(len(arg_input)),
        [alignutil.parse_constituency(constituency_predictor, p) for p in arg_input]
    ))
    # build the raw input df that the alignment and search algorithms build on top of...
    input_df_dict = {}
    for txt_id in output['parse_constituency']:
        tokens = []
        for token_i in range(len(output['parse_constituency'][txt_id]['tokens'])):
            tokens.append((
                output['parse_constituency'][txt_id]['tokens'][token_i],
                '',
                [output['parse_constituency'][txt_id]['pos_tags'][token_i]],
            ))
        input_df_dict[txt_id] = tokens
    input_df = pd.DataFrame(input_df_dict.values(), index=input_df_dict.keys())
    input_df = input_df.applymap(lambda x: ('', '', []) if (x is None) else x)
    input_df.columns = [f'txt{i}' for i in range(len(input_df.columns))]
    # align the texts!
    align_df = input_df.loc[[0]]
    for i in range(1, len(input_df)):
        align_df, align_df_score = alignutil.alignRowMajorLocal(
            align_df,
            input_df.loc[[i]],
            embed_model=fasttext
        )
    # convert the final alignment output to an outputtable format
    output['alignment'] = alignutil.alignment_to_jsondict(align_df)['alignment']
    output['alignment_rawtext'] = arg_input
    # build output
    return output


@app.route('/api/alignop/canshift', methods=['GET'])
def api_alignop_canshift():
    print('... called /api/alignop/canshift ...')
    # retrieve arguments
    try:
        arg_alignment = {'alignment': json.loads(request.args['alignment'])}
        arg_row = int(request.args['row'])
        arg_col = int(request.args['col'])
        arg_shiftdist = int(request.args['shift_dist'])
    except:
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback':f'{traceback.format_exc()}'
        }
    align_df = alignutil.jsondict_to_alignment(arg_alignment)
    result_canshift = alignutil.canShiftCells(
        align_df,
        shift_rows=[arg_row],
        shift_col=f'txt{arg_col}',
        shift_distance=arg_shiftdist,
        shift_size=1
    )
    return {'is_legal': result_canshift}


@app.route('/api/alignop/shift', methods=['GET'])
def api_alignop_shift():
    print('... called /api/alignop/shift ...')
    # retrieve arguments
    try:
        arg_alignment = {'alignment': json.loads(request.args['alignment'])}
        arg_row = int(request.args['row'])
        arg_col = int(request.args['col'])
        arg_shiftdist = int(request.args['shift_dist'])
    except:
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback':f'{traceback.format_exc()}'
        }
    align_df = alignutil.jsondict_to_alignment(arg_alignment)
    align_df = alignutil.shiftCells(
        align_df,
        shift_rows=[arg_row],
        shift_col=f'txt{arg_col}',
        shift_distance=arg_shiftdist,
    )
    return alignutil.alignment_to_jsondict(align_df)


@app.route('/api/alignscore', methods=['GET'])
def api_alignscore():
    print('... called /api/alignscore ...')
    # retrieve arguments
    try:
        arg_alignment = {'alignment': json.loads(request.args['alignment'])}
    except:
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback':f'{traceback.format_exc()}'
        }
    align_df = alignutil.jsondict_to_alignment(arg_alignment)
    singlescore, components, rawscores = alignutil.scoreAlignment(
        align_df,
        spacy_model=sp,
        scispacy_model=scisp,
        scispacy_linker=linker,
        embed_model=fasttext,
        # max_row_length=max_row_length,
    )
    return {'alignment_score': singlescore}
