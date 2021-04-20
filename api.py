import gensim
import allennlp_models.tagging
from allennlp.predictors.predictor import Predictor
from flask import Flask, request, jsonify, url_for
from celery import Celery
import traceback
import json
import time
from os import path
import pandas as pd
import numpy as np

import alignutil

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Flask-specific code...
app = Flask(__name__)

# Configure Celery...
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# NLP model imports...
print('=== STARTING NLP MODEL IMPORTS ===')
coref_predictor = None
constituency_predictor = None
dependency_predictor = None
fasttext = None
sp = None
scisp = None
linker = None

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


@celery.task(bind=True)
def task_textalign(self, arg_input):
    # initialize the progress variables
    rows_aligned = 0
    rows_total = len(arg_input)
    self.update_state(
        state='PROGRESS',
        meta={
            'current': rows_aligned,
            'total': rows_total,
            'status': 'Currently performing constituency parse...'
        }
    )
    # actually do some work now!
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
    rows_aligned += 1
    self.update_state(
        state='PROGRESS',
        meta={
            'current': rows_aligned,
            'total': rows_total,
            'status': f'Currently aligning... progress ({rows_aligned}/{rows_total})'
        }
    )
    # align the texts!
    align_df = input_df.loc[[0]]
    for i in range(1, len(input_df)):
        rows_aligned += 1
        align_df, align_df_score = alignutil.alignRowMajorLocal(
            align_df,
            input_df.loc[[i]],
            embed_model=fasttext
        )
        self.update_state(
            state='PROGRESS',
            meta={
                'current': rows_aligned,
                'total': rows_total,
                'status': f'Currently aligning... progress ({rows_aligned}/{rows_total})'
            }
        )
        # print(f'aligned {rows_aligned}/{rows_total}')
    # convert the final alignment output to an outputtable format
    output['alignment'] = alignutil.alignment_to_jsondict(align_df)['alignment']
    return output


@app.route('/status/textalign/<task_id>', methods=['GET'])
def taskstatus_textalign(task_id):
    # print('... called /status/textalign/<ID> ... ...')
    task = task_textalign.AsyncResult(task_id)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'parse_constituency' in task.info:
            response['parse_constituency'] = task.info['parse_constituency']
        if 'alignment' in task.info:
            response['alignment'] = task.info['alignment']
    else:
        # if we are in the failure state...
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


@app.route('/api/textalign', methods=['POST'])
def api_textalign():
    print('... called /api/textalign ...')
    # retrieve arguments
    request_args = request.get_json()
    try:
        # arg_input = request_args['input'].split('\n') if ('input' in request_args) else ['default']
        arg_input = [e.strip() for e in request_args['input'].split('\n') if e.strip()!='']
    except:
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback': f'{traceback.format_exc()}'
        }
    task = task_textalign.apply_async(kwargs={
        'arg_input':arg_input,
    })
    return jsonify({
        'location': url_for('taskstatus_textalign', task_id=task.id)
    }), 202


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
    return jsonify({'is_legal': result_canshift})


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
    return jsonify(alignutil.alignment_to_jsondict(align_df))


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
    return jsonify(alignutil.alignment_to_jsondict(align_df))


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
    return jsonify(alignutil.alignment_to_jsondict(align_df))


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
    return jsonify({'alignment_score': singlescore})


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
    # set some temporary model variable names...
    spacy_model = sp
    scispacy_model = scisp
    scispacy_linker = linker
    embed_model = fasttext
    max_row_length = None
    term_weight_func=None
    weight_components=None
    # do the greedy step search ----
    # calculate the step (alignment operation) space...
    valid_operations = []
    valid_operations += [('none', 0)]
    # add shift steps
    for col_i in range(len(align_df.columns)):
        # get all valid clumps of rows in the column
        col_texts = [
            e for e in zip([e[0]
            for e in align_df[align_df.columns[col_i]]], align_df.index)
            if len(e[0])!=0
        ]
        row_clumps = {}
        for col_word in set([e[0] for e in col_texts]):
            row_clumps[col_word] = [e[1] for e in col_texts if e[0]==col_word]
        # calculate all possible shifts for each clump of rows
        for row_clump_word in row_clumps:
            for distance in range(-1 * len(align_df.columns), len(align_df.columns)):
                if distance != 0 and alignutil.canShiftCells(align_df, row_clumps[row_clump_word], align_df.columns[col_i], distance, 1):
                    valid_operations += [
                        ('shift', row_clumps[row_clump_word], align_df.columns[col_i], distance, 1)
                    ]
    # print(valid_operations)
    # run through all of the operations and calculate what their result would be!
    candidates = []
    operation_i = 1
    for selected_operation in valid_operations:
        if selected_operation[0]=='shift':
            operated = alignutil.shiftCells(
                align_df,
                selected_operation[1],
                selected_operation[2],
                selected_operation[3],
                shift_size=selected_operation[4],
            )
        # elif selected_operation[0]=='split':
        #     operated = alignutil.splitCol(align_df, selected_operation[1], right_align=selected_operation[2])
        # elif selected_operation[0]=='merge':
        #     operated = alignutil.mergeCol(align_df, selected_operation[1])
        elif selected_operation[0]=='none':
            operated = align_df
        else:
            raise ValueError('uh oh, undefined operation')
        singlescore, components, rawscores = alignutil.scoreAlignment(
            operated,
            spacy_model=spacy_model,
            scispacy_model=scispacy_model, scispacy_linker=scispacy_linker,
            embed_model=embed_model,
            max_row_length=max_row_length,
            # weight_components=weight_components
        )
        candidates.append((operated, singlescore, selected_operation))
        print(f'computed {operation_i}/{len(valid_operations)} operations')
        operation_i += 1
    # sort the result candidates by score, descending
    candidates.sort(key=lambda x: -1 * x[1])
    # and pick the best candidate (operated, singlescore, selected_operation)
    greedystep_df, greedystep_score, greedystep_operation = candidates[0]
    print('greedy step chose', greedystep_operation)
    return jsonify(alignutil.alignment_to_jsondict(greedystep_df))
