import gensim
import allennlp_models.tagging
from allennlp.predictors.predictor import Predictor
from flask import Flask, request
from celery import Celery
import traceback
import json
from os import path
import pandas as pd
import numpy as np

import alignutil

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Flask-specific code...
app = Flask(__name__)

# # Configure Celery...
# app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
# app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
# celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# celery.conf.update(app.config)

# initialize the top-level variables to None lmao
coref_predictor = None
constituency_predictor = None
dependency_predictor = None
fasttext = None
sp = None
scisp = None
linker = None

@app.before_first_request
def before_first_request_setup():
    # NLP model imports...
    print('=== STARTING NLP MODEL IMPORTS ===')
    global coref_predictor
    global constituency_predictor
    global dependency_predictor
    global fasttext
    global sp
    global scisp
    global linker

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
    # import spacy
    # sp = spacy.load('en_core_web_sm')
    # import scispacy
    # from scispacy.linking import EntityLinker
    # scisp = spacy.load('en_core_sci_sm')
    # linker = scisp.add_pipe('scispacy_linker', config={'resolve_abbreviations': True, 'linker_name': 'umls'})

    print('=== FINISHED NLP MODEL IMPORTS ===')


@app.route('/api/textalign', methods=['POST'])
def api_textalign():
    print('... called /api/textalign ...')
    # retrieve arguments
    request_args = request.get_json()
    try:
        # arg_id = int(request_args['id'])
        # arg_input = request_args['input'].split('\n') if ('input' in request_args) else ['default']
        arg_input = [e.strip() for e in request_args['input'].split('\n') if e.strip()!='']
    except:
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback': f'{traceback.format_exc()}'
        }
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
        print(f'aligned {i+1}/{len(input_df)}')
    # convert the final alignment output to an outputtable format
    output['alignment'] = alignutil.alignment_to_jsondict(align_df)['alignment']
    return output


@app.route('/api/alignop/canshift', methods=['POST'])
def api_alignop_canshift():
    # retrieve arguments
    request_args = request.get_json()
    try:
        arg_alignment = {'alignment': json.loads(request_args['alignment'])}
        arg_row = int(request_args['row'])
        arg_col = int(request_args['col'])
        arg_shiftdist = int(request_args['shift_dist'])
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


@app.route('/api/alignop/shift', methods=['POST'])
def api_alignop_shift():
    print('... called /api/alignop/shift ...')
    # retrieve arguments
    request_args = request.get_json()
    try:
        arg_alignment = {'alignment': json.loads(request_args['alignment'])}
        arg_row = int(request_args['row'])
        arg_col = int(request_args['col'])
        arg_shiftdist = int(request_args['shift_dist'])
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


@app.route('/api/alignop/insertcol', methods=['POST'])
def api_alignop_insertcol():
    print('... called /api/alignop/insertcol ...')
    # retrieve arguments
    request_args = request.get_json()
    try:
        arg_alignment = {'alignment': json.loads(request_args['alignment'])}
        arg_col = int(request_args['col'])
        arg_insertafter = request_args['insertafter']
    except:
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback':f'{traceback.format_exc()}'
        }
    align_df = alignutil.jsondict_to_alignment(arg_alignment)
    align_df = alignutil.insertColumn(
        align_df,
        insert_col=f'txt{arg_col}',
        insert_after=arg_insertafter,
    )
    return alignutil.alignment_to_jsondict(align_df)


@app.route('/api/alignop/deletecol', methods=['POST'])
def api_alignop_deletecol():
    print('... called /api/alignop/deletecol ...')
    # retrieve arguments
    request_args = request.get_json()
    try:
        arg_alignment = {'alignment': json.loads(request_args['alignment'])}
        arg_col = int(request_args['col'])
    except:
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback':f'{traceback.format_exc()}'
        }
    align_df = alignutil.jsondict_to_alignment(arg_alignment)
    align_df = alignutil.deleteColumn(
        align_df,
        delete_col=f'txt{arg_col}',
    )
    return alignutil.alignment_to_jsondict(align_df)


@app.route('/api/alignscore', methods=['POST'])
def api_alignscore():
    print('... called /api/alignscore ...')
    # retrieve arguments
    request_args = request.get_json()
    try:
        arg_alignment = {'alignment': json.loads(request_args['alignment'])}
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


@app.route('/api/alignsearch', methods=['POST'])
def api_alignsearch():
    print('... called /api/alignsearch ...')
    # retrieve arguments
    request_args = request.get_json()
    try:
        arg_alignment = {'alignment': json.loads(request_args['alignment'])}
    except:
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback':f'{traceback.format_exc()}'
        }
    align_df = alignutil.jsondict_to_alignment(arg_alignment)
    greedystep_df, greedystep_score, greedystep_operation = alignutil.searchGreedyStep(
        align_df,
        spacy_model=sp,
        scispacy_model=scisp,
        scispacy_linker=linker,
        embed_model=fasttext,
        # max_row_length=max_row_length,
    )
    print('greedy step chose', greedystep_operation)
    return alignutil.alignment_to_jsondict(greedystep_df)
