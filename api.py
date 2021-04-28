import gensim
import allennlp_models.tagging
from allennlp.predictors.predictor import Predictor
from flask import Flask, request, jsonify, url_for
from celery import Celery
import traceback
import json
import time
from os import path
import itertools
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
            'status': 'Currently performing constituency parses...'
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
            'traceback': f'{traceback.format_exc()}'
        }
    align_df = alignutil.jsondict_to_alignment(arg_alignment)
    try:
        align_df = alignutil.shiftCells(
            align_df,
            shift_rows=[arg_row],
            shift_col=f'txt{arg_col}',
            shift_distance=arg_shiftdist,
        )
    except:
        # if shifting fails, just don't do it
        pass
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


@celery.task(bind=True)
def task_alignsearch(self, arg_alignment, arg_alignment_cols_locked):
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
    self.update_state(
        state='PROGRESS',
        meta={
            'current': 0,
            'total': 0,
            'status': 'Currently calculating operation space...'
        }
    )
    # calculate the step (alignment operation) space...
    valid_operations = []
    valid_operations += [('none', 0)]
    # add shift steps
    for col_i in range(len(align_df.columns)):
        # only shift from this col if it is not locked...
        if not arg_alignment_cols_locked[col_i]:
            # get all valid clumps of rows in the column
            col_texts = [
                e for e in zip([e[0]
                for e in align_df[align_df.columns[col_i]]], align_df.index)
                if len(e[0])!=0
            ]
            row_clumps = {}
            for col_word in set([e[0] for e in col_texts]):
                row_clumps[col_word] = [e[1] for e in col_texts if e[0]==col_word]
            # set how many columns we are shifting at once
            shift_size = 1
            # establish the basic shift ranges, expand it later
            shift_lower_bound = 0
            shift_upper_bound = 0
            # now add locked column check info to the shift range
            print(f'{arg_alignment_cols_locked[:col_i]}, {arg_alignment_cols_locked[col_i+1:]}')
            if (col_i > 0) and not arg_alignment_cols_locked[col_i-1]:
                shift_lower_bound = -1 * min(
                    col_i,
                    [sum(1 for _ in group) for e, group in itertools.groupby(arg_alignment_cols_locked[:col_i])][-1]
                )
            if (col_i < len(arg_alignment_cols_locked)-1) and not arg_alignment_cols_locked[col_i+1]:
                shift_upper_bound = min(
                    len(align_df.columns) - col_i,
                    [sum(1 for _ in group) for e, group in itertools.groupby(arg_alignment_cols_locked[col_i+1:])][0]
                ) - shift_size + 1
            print(f'col-lock shift bounds of txt{col_i}: {shift_lower_bound}, {shift_upper_bound}')
            # now calculate all legal shifts :)
            for distance in range(shift_lower_bound, shift_upper_bound):
                # calculate legality of shifting for each clump of rows
                for row_clump_word in row_clumps:
                    if distance != 0 and alignutil.canShiftCells(align_df, row_clumps[row_clump_word], align_df.columns[col_i], distance, shift_size):
                        valid_operations += [
                            ('shift', row_clumps[row_clump_word], align_df.columns[col_i], distance, shift_size)
                        ]
    # initialize the progress variables
    states_calculated = 0
    states_total = len(valid_operations)
    self.update_state(
        state='PROGRESS',
        meta={
            'current': states_calculated,
            'total': states_total,
            'status': f'Currently calculating operation scores... progress ({states_calculated}/{states_total})'
        }
    )
    # run through all of the operations and calculate what their result would be!
    candidates = []
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
        states_calculated += 1
        self.update_state(
            state='PROGRESS',
            meta={
                'current': states_calculated,
                'total': states_total,
                'status': f'Currently calculating operation scores... progress ({states_calculated}/{states_total})'
            }
        )
    # sort the result candidates by score, descending
    candidates.sort(key=lambda x: -1 * x[1])
    # and pick the best candidate (operated, singlescore, selected_operation)
    greedystep_df, greedystep_score, greedystep_operation = candidates[0]
    print('greedy step chose', greedystep_operation)
    # generate a nice readable status text
    status_text = 'No operation performed'
    if (greedystep_operation[0]=='shift') and (greedystep_operation[3]!=0):
        status_text = f'Shifted {greedystep_operation[4]} cells(s)'
        status_text += f' starting from column {greedystep_operation[2]}'
        status_text += f' in rows {greedystep_operation[1]}'
        if greedystep_operation[3]>0:
            status_text += f' by {greedystep_operation[3]} cell(s) to the right'
        else:
            status_text += f' by {-1*greedystep_operation[3]} cell(s) to the left'
    return {
        'status': status_text,
        'alignment': alignutil.alignment_to_jsondict(greedystep_df)['alignment']
    }


@app.route('/status/alignsearch/<task_id>', methods=['GET'])
def taskstatus_alignsearch(task_id):
    # print('... called /status/alignsearch/<ID> ... ...')
    task = task_alignsearch.AsyncResult(task_id)
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


@app.route('/api/alignsearch', methods=['POST'])
def api_alignsearch():
    # TODO refactor into celery task
    print('... called /api/alignsearch ...')
    # retrieve arguments
    request_args = request.get_json()
    try:
        arg_alignment = {'alignment': json.loads(request_args['alignment'])}
        arg_alignment_cols_locked = json.loads(request_args['alignment_cols_locked'])
    except:
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback':f'{traceback.format_exc()}'
        }
    task = task_alignsearch.apply_async(kwargs={
        'arg_alignment':arg_alignment,
        'arg_alignment_cols_locked':arg_alignment_cols_locked,
    })
    return jsonify({
        'location': url_for('taskstatus_alignsearch', task_id=task.id)
    }), 202
