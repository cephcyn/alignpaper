import gensim
import allennlp_models.tagging
from allennlp.predictors.predictor import Predictor
from flask import Flask, request, jsonify, url_for
from celery import Celery
import traceback
import json
import time
import random
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
def task_textalign(self, arg_input, arg_score_components):
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
    # compute max_row_length to be used for this set of texts
    max_row_length = alignutil.maxRowLength(align_df)
    output['alignment_max_row_length'] = max_row_length
    # convert the final alignment output to an outputtable format
    output['alignment'] = alignutil.alignment_to_jsondict(align_df)['alignment']
    # get alignment score
    singlescore, components, rawscores = alignutil.scoreAlignment(
        align_df,
        spacy_model=sp,
        scispacy_model=scisp,
        scispacy_linker=linker,
        embed_model=fasttext,
        max_row_length=max_row_length,
        weight_components=arg_score_components,
    )
    output['alignment_score'] = singlescore
    output['alignment_score_components'] = list(components)
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
        if 'alignment_max_row_length' in task.info:
            response['alignment_max_row_length'] = task.info['alignment_max_row_length']
        if 'alignment_score' in task.info:
            response['alignment_score'] = task.info['alignment_score']
        if 'alignment_score_components' in task.info:
            response['alignment_score_components'] = task.info['alignment_score_components']
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
        arg_input = [e.strip() for e in request_args['input'].split('\n') if e.strip()!='']
        arg_score_components = [float(e) for e in request_args['param_score_components']]
    except:
        print(traceback.format_exc())
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback': f'{traceback.format_exc()}'
        }
    task = task_textalign.apply_async(kwargs={
        'arg_input':arg_input,
        'arg_score_components':arg_score_components
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
        print(traceback.format_exc())
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
        arg_max_row_length = int(request_args['alignment_max_row_length'])
        arg_row = int(request_args['row'])
        arg_col = int(request_args['col'])
        arg_shiftdist = int(request_args['shift_dist'])
        arg_score_components = [float(e) for e in request_args['param_score_components']]
    except:
        print(traceback.format_exc())
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
            force_push=True,
        )
    except:
        # if shifting fails, just don't do it
        print(traceback.format_exc())
        pass
    output = alignutil.alignment_to_jsondict(align_df)
    # get alignment score
    singlescore, components, rawscores = alignutil.scoreAlignment(
        align_df,
        spacy_model=sp,
        scispacy_model=scisp,
        scispacy_linker=linker,
        embed_model=fasttext,
        max_row_length=arg_max_row_length,
        weight_components=arg_score_components,
    )
    output['alignment_max_row_length'] = alignutil.maxRowLength(align_df)
    output['alignment_score'] = singlescore
    output['alignment_score_components'] = list(components)
    return jsonify(output)


@app.route('/api/alignop/insertcol', methods=['POST'])
def api_alignop_insertcol():
    print('... called /api/alignop/insertcol ...')
    # retrieve arguments
    request_args = request.get_json()
    try:
        arg_alignment = {'alignment': json.loads(request_args['alignment'])}
        arg_max_row_length = int(request_args['alignment_max_row_length'])
        arg_col = int(request_args['col'])
        arg_insertafter = request_args['insertafter']
        arg_score_components = [float(e) for e in request_args['param_score_components']]
    except:
        print(traceback.format_exc())
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
    output = alignutil.alignment_to_jsondict(align_df)
    # get alignment score
    singlescore, components, rawscores = alignutil.scoreAlignment(
        align_df,
        spacy_model=sp,
        scispacy_model=scisp,
        scispacy_linker=linker,
        embed_model=fasttext,
        max_row_length=arg_max_row_length,
        weight_components=arg_score_components,
    )
    output['alignment_max_row_length'] = alignutil.maxRowLength(align_df)
    output['alignment_score'] = singlescore
    output['alignment_score_components'] = list(components)
    return jsonify(output)


@app.route('/api/alignop/deletecol', methods=['POST'])
def api_alignop_deletecol():
    print('... called /api/alignop/deletecol ...')
    # TODO should it be legal to delete a column when there is text inside?
    # retrieve arguments
    request_args = request.get_json()
    try:
        arg_alignment = {'alignment': json.loads(request_args['alignment'])}
        arg_max_row_length = int(request_args['alignment_max_row_length'])
        arg_col = int(request_args['col'])
        arg_score_components = [float(e) for e in request_args['param_score_components']]
    except:
        print(traceback.format_exc())
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback':f'{traceback.format_exc()}'
        }
    align_df = alignutil.jsondict_to_alignment(arg_alignment)
    align_df = alignutil.deleteColumn(
        align_df,
        delete_col=f'txt{arg_col}',
    )
    output = alignutil.alignment_to_jsondict(align_df)
    # get alignment score
    singlescore, components, rawscores = alignutil.scoreAlignment(
        align_df,
        spacy_model=sp,
        scispacy_model=scisp,
        scispacy_linker=linker,
        embed_model=fasttext,
        max_row_length=arg_max_row_length,
        weight_components=arg_score_components
    )
    output['alignment_max_row_length'] = alignutil.maxRowLength(align_df)
    output['alignment_score'] = singlescore
    output['alignment_score_components'] = list(components)
    return jsonify(output)


@app.route('/api/alignop/mergecol', methods=['POST'])
def api_alignop_mergecol():
    print('... called /api/alignop/mergecol ...')
    # retrieve arguments
    request_args = request.get_json()
    try:
        arg_alignment = {'alignment': json.loads(request_args['alignment'])}
        arg_max_row_length = int(request_args['alignment_max_row_length'])
        arg_col = int(request_args['col'])
        arg_score_components = [float(e) for e in request_args['param_score_components']]
    except:
        print(traceback.format_exc())
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback':f'{traceback.format_exc()}'
        }
    align_df = alignutil.jsondict_to_alignment(arg_alignment)
    align_df = alignutil.mergeColumn(
        align_df,
        merge_col=f'txt{arg_col}',
    )
    output = alignutil.alignment_to_jsondict(align_df)
    # get alignment score
    singlescore, components, rawscores = alignutil.scoreAlignment(
        align_df,
        spacy_model=sp,
        scispacy_model=scisp,
        scispacy_linker=linker,
        embed_model=fasttext,
        max_row_length=arg_max_row_length,
        weight_components=arg_score_components
    )
    output['alignment_max_row_length'] = alignutil.maxRowLength(align_df)
    output['alignment_score'] = singlescore
    output['alignment_score_components'] = list(components)
    return jsonify(output)


@app.route('/api/alignop/splitsinglecol', methods=['POST'])
def api_alignop_splitsinglecol():
    print('... called /api/alignop/splitsinglecol ...')
    # retrieve arguments
    request_args = request.get_json()
    try:
        arg_alignment = {'alignment': json.loads(request_args['alignment'])}
        arg_max_row_length = int(request_args['alignment_max_row_length'])
        arg_col = int(request_args['col'])
        arg_right_align = bool(request_args['right_align'])
        arg_score_components = [float(e) for e in request_args['param_score_components']]
    except:
        print(traceback.format_exc())
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback':f'{traceback.format_exc()}'
        }
    align_df = alignutil.jsondict_to_alignment(arg_alignment)
    align_df = alignutil.splitSingleColumn(
        align_df,
        split_col=f'txt{arg_col}',
        right_align=arg_right_align,
    )
    output = alignutil.alignment_to_jsondict(align_df)
    # get alignment score
    singlescore, components, rawscores = alignutil.scoreAlignment(
        align_df,
        spacy_model=sp,
        scispacy_model=scisp,
        scispacy_linker=linker,
        embed_model=fasttext,
        max_row_length=arg_max_row_length,
        weight_components=arg_score_components
    )
    output['alignment_max_row_length'] = alignutil.maxRowLength(align_df)
    output['alignment_score'] = singlescore
    output['alignment_score_components'] = list(components)
    return jsonify(output)


@app.route('/api/alignop/splittriecol', methods=['POST'])
def api_alignop_splittriecol():
    print('... called /api/alignop/splittriecol ...')
    # retrieve arguments
    request_args = request.get_json()
    try:
        arg_alignment = {'alignment': json.loads(request_args['alignment'])}
        arg_max_row_length = int(request_args['alignment_max_row_length'])
        arg_col = int(request_args['col'])
        arg_right_align = bool(request_args['right_align'])
        arg_score_components = [float(e) for e in request_args['param_score_components']]
    except:
        print(traceback.format_exc())
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback':f'{traceback.format_exc()}'
        }
    align_df = alignutil.jsondict_to_alignment(arg_alignment)
    align_df = alignutil.splitTrieColumn(
        align_df,
        split_col=f'txt{arg_col}',
        right_align=arg_right_align,
    )
    output = alignutil.alignment_to_jsondict(align_df)
    # get alignment score
    singlescore, components, rawscores = alignutil.scoreAlignment(
        align_df,
        spacy_model=sp,
        scispacy_model=scisp,
        scispacy_linker=linker,
        embed_model=fasttext,
        max_row_length=arg_max_row_length,
        weight_components=arg_score_components
    )
    output['alignment_max_row_length'] = alignutil.maxRowLength(align_df)
    output['alignment_score'] = singlescore
    output['alignment_score_components'] = list(components)
    return jsonify(output)


@app.route('/api/alignscore', methods=['POST'])
def api_alignscore():
    print('... called /api/alignscore ...')
    # retrieve arguments
    request_args = request.get_json()
    try:
        arg_alignment = {'alignment': json.loads(request_args['alignment'])}
        arg_max_row_length = int(request_args['alignment_max_row_length'])
        arg_score_components = [float(e) for e in request_args['param_score_components']]
    except:
        print(traceback.format_exc())
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
        max_row_length=arg_max_row_length,
        weight_components=arg_score_components
    )
    output = {}
    output['alignment_score'] = singlescore
    output['alignment_score_components'] = list(components)
    return jsonify(output)


@celery.task(bind=True)
def task_alignsearch(
        self,
        arg_alignment,
        arg_max_row_length,
        arg_alignment_cols_locked,
        arg_greedysteps,
        arg_score_components,
        arg_move_distrib,
        arg_none_optimal_cutoff):
    align_df = alignutil.jsondict_to_alignment(arg_alignment)
    # set some temporary variable names...
    spacy_model = sp
    scispacy_model = scisp
    scispacy_linker = linker
    embed_model = fasttext
    max_row_length = arg_max_row_length
    term_weight_func = None
    weight_components = None
    move_distrib = arg_move_distrib
    none_optimal_cutoff = arg_none_optimal_cutoff
    # initialize move selection resources
    random.seed()
    none_optimal_n = 0
    # function to select a move from a given move distribution
    def select_move(move_distrib):
        move_distrib_sum = sum(e[1] for e in move_distrib)
        move_distrib_acc = [
            (move_distrib[i][0], sum([e[1] for e in move_distrib[:i]]))
            for i in range(len(move_distrib))
        ]
        move_i = random.randint(0, move_distrib_sum-1)
        return [e for e in move_distrib_acc if e[1]<=move_i][-1][0]
    # initialize some history tracking variables
    operation_history = []
    initial_singlescore, initial_components, initial_rawscores = alignutil.scoreAlignment(
        align_df,
        spacy_model=sp,
        scispacy_model=scisp,
        scispacy_linker=linker,
        embed_model=fasttext,
        max_row_length=max_row_length,
        weight_components=arg_score_components,
    )
    optimal_score = initial_singlescore
    optimal_scorecomponents = initial_components
    optimal_df = align_df
    optimal_step_i = 0
    # now actually do the search process, take all the steps we need
    for step_number in range(arg_greedysteps):
        self.update_state(
            state='PROGRESS',
            meta={
                'current': step_number,
                'total': arg_greedysteps,
                'status': f'Step {step_number+1}/{arg_greedysteps}: calculating operation space...'
            }
        )
        # calculate the step (alignment operation) space...
        valid_operations = []
        # add none step
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
                # now calculate all legal shifts :)
                for distance in range(shift_lower_bound, shift_upper_bound+1):
                    # calculate legality of shifting for each clump of rows
                    for row_clump_word in row_clumps:
                        if distance != 0 and alignutil.canShiftCells(align_df, row_clumps[row_clump_word], align_df.columns[col_i], distance, shift_size):
                            valid_operations += [
                                ('shift', row_clumps[row_clump_word], align_df.columns[col_i], distance, shift_size)
                            ]
        # # add merge steps
        # valid_operations += [('merge', e) for e in align_df.columns[:-1]]
        # initialize the progress variables
        states_calculated = 0
        states_total = len(valid_operations)
        print(valid_operations)
        self.update_state(
            state='PROGRESS',
            meta={
                'current': step_number,
                'total': arg_greedysteps,
                'status': f'Step {step_number+1}/{arg_greedysteps}: calculating operation scores (progress {states_calculated}/{states_total})'
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
            elif selected_operation[0]=='merge':
                operated = alignutil.mergeColumn(align_df, selected_operation[1])
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
                weight_components=arg_score_components,
            )
            candidates.append((operated, singlescore, selected_operation, components))
            states_calculated += 1
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': step_number,
                    'total': arg_greedysteps,
                    'status': f'Step {step_number+1}/{arg_greedysteps}: calculating operation scores (progress {states_calculated}/{states_total})'
                }
            )
        # sort the result candidates by score, descending
        candidates.sort(key=lambda x: -1 * x[1])
        # keep track of how many times in a row the best option has been 'No move'
        if candidates[0][2][0]=='none':
            none_optimal_n += 1
        else:
            none_optimal_n = 0
        # now actually make a step
        move = select_move(move_distrib)
        if move == 'greedy':
            # pick the candidate with best score
            step_df, step_score, step_operation, step_scorecomponents = candidates[0]
            print(f'greedy step chose {step_operation} with score {step_score}')
        elif move == 'randomwalk':
            selected = random.randint(0, len(candidates)-1)
            step_df, step_score, step_operation, step_scorecomponents = candidates[selected]
            if step_operation=='None':
                # randomwalk isn't allowed to take 'None' move :P
                selected = random.randint(0, len(candidates)-1)
                step_df, step_score, step_operation, step_scorecomponents = candidates[selected]
            print(f'random step chose {step_operation} with score {step_score}')
        # check if this step is the new optimal step
        if step_score > optimal_score:
            optimal_score = step_score
            optimal_scorecomponents = step_scorecomponents
            optimal_df = step_df
            optimal_step_i = step_number
        # generate a nice readable status text
        status_text = f'{move} - '
        if (step_operation[0]=='shift') and (step_operation[3]!=0):
            status_text += f'Shifted {step_operation[4]} cells(s)'
            status_text += f' starting from column {step_operation[2]}'
            status_text += f' in rows {step_operation[1]}'
            if step_operation[3]>0:
                status_text += f' by {step_operation[3]} cell(s) to the right'
            else:
                status_text += f' by {-1*step_operation[3]} cell(s) to the left'
        elif step_operation[0]=='merge':
            status_text += f'Merged {step_operation[1]} with column to the right'
        else:
            status_text += 'No operation performed'
        status_text += f' (score is now {step_score})'
        status_text += f' (subscores are {step_scorecomponents})'
        operation_history.append(status_text)
        # break out of this loop if we have hit the limit on # of none-optimals
        if none_optimal_n >= none_optimal_cutoff:
            print(f'breaking step loop because None operation was optimal {none_optimal_n}x in a row')
            break
        # set align_df to step_df to ready for next greedy step
        align_df = step_df
    # only keep the slice of operation history up until the optimal state
    operation_history = operation_history[:optimal_step_i+1]
    # prepend initial score info to the status
    operation_history = [
        f'Initial alignment score {initial_singlescore} (components {initial_components})'
    ] + operation_history
    # clean up operation_history to have step numbers
    operation_history = [
        f'({i}/{len(operation_history)-1}): {operation_history[i]}'
        for i in range(len(operation_history))
    ]
    return {
        'status': '\n'.join(operation_history),
        'alignment': alignutil.alignment_to_jsondict(optimal_df)['alignment'],
        'alignment_max_row_length': alignutil.maxRowLength(align_df),
        'alignment_score': optimal_score,
        'alignment_score_components': list(optimal_scorecomponents)
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
        if 'alignment_max_row_length' in task.info:
            response['alignment_max_row_length'] = task.info['alignment_max_row_length']
        if 'alignment_score' in task.info:
            response['alignment_score'] = task.info['alignment_score']
        if 'alignment_score_components' in task.info:
            response['alignment_score_components'] = task.info['alignment_score_components']
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
    print('... called /api/alignsearch ...')
    # retrieve arguments
    request_args = request.get_json()
    try:
        arg_alignment = {'alignment': json.loads(request_args['alignment'])}
        arg_max_row_length = int(request_args['alignment_max_row_length'])
        arg_alignment_cols_locked = json.loads(request_args['alignment_cols_locked'])
        arg_greedysteps = int(json.loads(request_args['greedysteps']))
        arg_score_components = [float(e) for e in request_args['param_score_components']]
        arg_move_distrib = [int(e) for e in request_args['param_move_distrib']]
        arg_none_optimal_cutoff = int(json.loads(request_args['param_search_cutoff']))
        #if ('param_move_distrib' in request_args)
    except:
        print(traceback.format_exc())
        return {
            'error': 'improperly formatted or missing arguments',
            'traceback':f'{traceback.format_exc()}'
        }
    task = task_alignsearch.apply_async(kwargs={
        'arg_alignment':arg_alignment,
        'arg_max_row_length':arg_max_row_length,
        'arg_alignment_cols_locked':arg_alignment_cols_locked,
        'arg_greedysteps':arg_greedysteps,
        'arg_score_components':arg_score_components,
        'arg_move_distrib':[('greedy', arg_move_distrib[0]), ('randomwalk', arg_move_distrib[1])],
        'arg_none_optimal_cutoff':arg_none_optimal_cutoff,
    })
    return jsonify({
        'location': url_for('taskstatus_alignsearch', task_id=task.id)
    }), 202
