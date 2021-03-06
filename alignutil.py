import warnings
import math
import traceback
from nltk.metrics import edit_distance
import numpy as np
import pandas as pd

# imports which are more hesitant at this point
import gensim
import spacy
import scispacy

# Get the constituency parses of a text input
# also adds span info to each constituency chunk for ease of later reference (?)
# Guide on constituency parses:
# https://web.stanford.edu/~jurafsky/slp3/13.pdf
# POS tag info:
# https://universaldependencies.org/u/pos/
# https://cs.nyu.edu/grishman/jet/guide/PennPOS.html
# http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
# TODO-REFERENCE originally from analyze.ipynb
def parse_constituency(constituency_predictor, phrase):
    # Compute the actual constituency parse
    p = constituency_predictor.predict(
        sentence=' '.join(phrase.split()).strip()
    )
    # Edit the constituency parse dict to contain a 'spans' field in the hierplane_tree
    # recursive function that this makes use of...
    # Side effect: modifies the given node to add a 'spans' field
    def add_constituency_span_recursive(node):
        curr_span_start = node['spans'][0]['start']
        curr_word = node['word']
        for c in (node['children'] if ('children' in node) else []):
            c_start = curr_span_start + curr_word.index(c['word'])
            c_end = c_start + len(c['word'])
            c['spans'] = [{
                'start': c_start,
                'end': c_end,
                'start-char': c_start,
                'end-char': c_end
            }]
            add_constituency_span_recursive(c)
            curr_span_start += len(c['word'])+1
            curr_word = curr_word[len(c['word'])+1:]
        return
    p['hierplane_tree']['root']['spans'] = [{
        'start': 0,
        'end': len(p['hierplane_tree']['root']['word']),
        'start-char': 0,
        'end-char': len(p['hierplane_tree']['root']['word'])
    }]
    add_constituency_span_recursive(p['hierplane_tree']['root'])
    # remove elements of the tree that don't get used at all
    p.pop('class_probabilities', None)
    p.pop('spans', None)
    p.pop('num_spans', None)
    p['hierplane_tree'].pop('linkNameToLabel', None)
    p['hierplane_tree'].pop('nodeTypeToStyle', None)
    return p


# Parse JSONable dict format into alignment DF
# TODO-REFERENCE originally from alignment.ipynb
def jsondict_to_alignment(alignment_dict):
    rows = {}
    for row in alignment_dict['alignment']:
        rows[row['id']] = [
            (
                ' '.join(row['txt'][i]) if ('txt' in row) else '',
                ' '.join(row['ppos'][i]) if ('ppos' in row) else '', # TODO haven't nailed down format... phrasePOS is hardly used
                row['pos'][i] if ('pos' in row) else []
            )
            for i in range(len(row['txt']))
        ]
    output_df = pd.DataFrame(rows.values(), index=rows.keys())
    output_df.columns = [f'txt{i}' for i in range(len(output_df.columns))]
    return output_df


# Parse alignment DF into JSONable dict format
# TODO-REFERENCE originally from alignment.ipynb
def alignment_to_jsondict(alignment_df):
    rows = []
    for index, row in alignment_df.iterrows():
        row_obj = {}
        row_obj['id'] = index
        row_obj['pos'] = []
        row_obj['txt'] = []
        for col in row:
            row_obj['pos'].append(col[2])
            row_obj['txt'].append([e for e in col[0].split(' ') if (e != '')])
        rows.append(row_obj)
    return {'alignment':rows}


# Extract tuple data from alignment DF
# TODO-REFERENCE originally from alignment.ipynb
def extractTup(data, tup_i=0, is_frame=True):
    types = {
        'segment': 0,
        'pos': 1,
        'cpos': 2
    }
    if tup_i in types:
        tup_i = types[tup_i]
    else:
        raise ValueError(f'tup_i not in types: {types.keys()}')
    if is_frame:
        return data.applymap(lambda x: x[tup_i])
    else:
        return data.map(lambda x: x[tup_i])


# Delete columns which are empty in all rows from alignment DF
# TODO-REFERENCE originally from alignment.ipynb
def removeEmptyColumns(align_df):
    output_columns = []
    for c in align_df.columns:
        align_df_c = extractTup(align_df.loc[:, c], tup_i='segment', is_frame=False)
        if len([e for e in align_df_c if e.strip() != '']) != 0:
            output_columns.append(c)
    output_df = align_df[output_columns]
    output_df.columns = [f'txt{i}' for i in range(len(output_df.columns))]
    return output_df


# Get the embedding of a phrase
# TODO-REFERENCE originally from alignment.ipynb
def get_phrase_embed(embed_model, phrase, remove_label=False, norm_zero_threshold=0.000000001):
    # split the phrase into tokens to pass into the embed model
    try:
        phraseS = phrase.split()
    except:
        return pd.DataFrame()
    # TODO remove stopwords?
    # retrieve the embeddings of each token in the phrase
    unknowns = []
    emb = []
    for w in phraseS:
        try:
            emb.append(embed_model[w])
        except:
            unknowns.append(w)
    # normalize each embed so that it has a norm of 1
    emb_normalized = []
    for i in range(len(emb)):
        e = emb[i]
        e_norm = np.linalg.norm(e)
        if e_norm < norm_zero_threshold:
            warnings.warn(f'embed vector for word \'{phraseS[i]}\' with extremely low norm value')
        emb_normalized.append(e / e_norm)
    emb = emb_normalized
    # if there are no recognized tokens in the phrase, return empty (same as non-splittable phrase)
    if len(emb) == 0:
        return pd.DataFrame()
    # Average the embeds for tokens which have embeds
    emb_avg = pd.DataFrame(emb).sum() / len(emb)
    if not remove_label:
        emb_avg['word'] = phrase
    return pd.DataFrame([emb_avg])


# Create an alignment of two given alignment DFs
# TODO-REFERENCE originally from alignment.ipynb
def alignRowMajorLocal(align_a, align_b, embed_model, use_types=False, remove_empty_cols=True, debug_print=False):
    # An implementation of Smith-Waterman alignment
    # RETURNS:
    #  1. The alignment DataFrame
    #  2. The S-W score associated with this alignment
    if remove_empty_cols:
        align_a = removeEmptyColumns(align_a)
        align_b = removeEmptyColumns(align_b)
    align_a_segment = extractTup(align_a, tup_i='segment')
    align_b_segment = extractTup(align_b, tup_i='segment')
    align_a_type = extractTup(align_a, tup_i='pos')
    align_b_type = extractTup(align_b, tup_i='pos')
    align_a_ctype = extractTup(align_a, tup_i='cpos')
    align_b_ctype = extractTup(align_b, tup_i='cpos')
    # Doing a general alignment
    align_a_elems = [i for i in range(len(align_a.columns))]
    align_b_elems = [i for i in range(len(align_b.columns))]
    if debug_print:
        print(align_a_elems)
        print(align_b_elems)
        print()
    def getScoreAligningIndices(index_a, index_b, embed_model):
        # A higher score is better / more match!
        # make sure all the segment texts are precomputed lol
        text_a = list(align_a_segment[align_a.columns[index_a]])
        text_b = list(align_b_segment[align_b.columns[index_b]])
        # TODO clean up this embed checking thing and remove the need for cached_phrase_embeds
        # TODO also unify this with the col embed variation measure somehow?
        cached_phrase_embeds = {}
        for text in text_a+text_b:
            if text not in cached_phrase_embeds:
                try:
                    cached_phrase_embeds[text] = get_phrase_embed(embed_model, text).drop('word', 1)
                except KeyError:
                    pass
        # start off with phrase embedding distance (current max is 60 for perfect match)
        # if we have embeds for any word in each set, ignore others and just use words we have embeds for
        if any(s in cached_phrase_embeds for s in text_a)\
                and any(s in cached_phrase_embeds for s in text_b):
            # calculate overall embeds
            embed_a = pd.concat([cached_phrase_embeds[text] for text
                                 in text_a if text in cached_phrase_embeds]).apply(lambda x: x.mean())
            embed_b = pd.concat([cached_phrase_embeds[text] for text
                                 in text_b if text in cached_phrase_embeds]).apply(lambda x: x.mean())
            # TODO can tweak this scoring calculation a little for performance
            score = 10 * (6 - np.linalg.norm(embed_a-embed_b))
        else:
            # use levenshtein dist as fallback... if either set has NO words with embeds available
            scaled_edits_sum = 0
            for phrase_a in [p for p in text_a if len(p) != 0]:
                for phrase_b in [p for p in text_b if len(p) != 0]:
                    # normalize to be in the range of [0,1]
                    # ref https://stackoverflow.com/questions/45783385/normalizing-the-edit-distance
                    scaled_edits_sum += edit_distance(phrase_a, phrase_b) / max(len(phrase_a), len(phrase_b))
            score = 60 * (1 - (scaled_edits_sum / (len(text_a) * len(text_b))))
        # add a component based on phrase type if that flag is set
        # TODO improve this?; this currently just returns -inf if mismatch of type sets
        # Might want to add support for aligning different types of phrase together...
        if use_types:
            # reduce to set
            types_a = set([t for t in align_a_type[align_a.columns[index_a]] if t.strip() != ''])
            types_b = set([t for t in align_b_type[align_b.columns[index_b]] if t.strip() != ''])
#             # check if we are handling a hard pos match
#             if any([((p in types_a) or (p in types_b)) for p in pos_must_match]):
            if len(types_a) != 0 and len(types_b) != 0 and types_a != types_b:
                score = -1 * math.inf
        # TODO: add a component based on phrase ctype (phrase POS breakdown) (?)
        if debug_print:
            print(f'scoring between '
                  +f'"{list(align_a_segment[align_a.columns[index_a]])}" and '
                  +f'"{list(align_b_segment[align_b.columns[index_b]])}": {score}')
        return score
    def getGapPenalty(length, size=1):
        return -1 * (1 * min(length,1) + 0.1 * max(length-1,0)) #* (1 + math.log(size))
    # Build score matrix of size (a-alignables + 1)x(b-alignables + 1)
    scores = np.zeros((len(align_a_elems)+1, len(align_b_elems)+1))
    # Build traceback matrix
    # traceback = 0 for end, 4 for W, 7 for NW, 9 for N (to calculate traceback, t%2 is N-ness, t%3 is W-ness)
    traceback = np.zeros((len(align_a_elems)+1, len(align_b_elems)+1))
    # Iterate through all of the cells to populate both the score and traceback matrices
    for i in range(1, scores.shape[0]):
        for j in range(1, scores.shape[1]):
            score_map = {}
            # calculate score for aligning nouns a[i] and b[j]
            score_map[
                scores[i-1,j-1] + getScoreAligningIndices(align_a_elems[i-1], align_b_elems[j-1], embed_model)
            ] = 7
            # calculate score for gap in i
            for i_gap in range(1, i):
                igap_score = scores[i-i_gap,j] + getGapPenalty(i_gap, size=len(align_a_elems))
                score_map[igap_score] = 9
            # calculate score for gap in j
            for j_gap in range(1, j):
                jgap_score = scores[i,j-j_gap] + getGapPenalty(j_gap, size=len(align_b_elems))
                score_map[jgap_score] = 4
            # add the possibility for unrelatedness
            score_map[0] = 0
            scores[i,j] = max(score_map.keys())
            traceback[i,j] = score_map[max(score_map.keys())]
    if debug_print:
        print()
        print(scores)
        print(traceback)
        print()
    # Do traceback to build our final alignment
    tracepoint = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
    points_a = []
    points_b = []
    while traceback[tracepoint] != 0:
        # contribute to the align information
        if traceback[tracepoint] == 7:
            # this is a point where two elements were aligned
            points_a.append(align_a_elems[tracepoint[0]-1])
            points_b.append(align_b_elems[tracepoint[1]-1])
        elif traceback[tracepoint] == 4:
            # this is a point where there was a gap inserted for row_a
            points_a.append(-1)
            points_b.append(align_b_elems[tracepoint[1]-1])
        elif traceback[tracepoint] == 9:
            # this is a point where there was a gap inserted for row_b
            points_a.append(align_a_elems[tracepoint[0]-1])
            points_b.append(-1)
        # step backwards
        tracepoint = (
            tracepoint[0] - int(traceback[tracepoint] % 2),
            tracepoint[1] - int(traceback[tracepoint] % 3))
    points_a = list(reversed(points_a))
    points_b = list(reversed(points_b))
    if len(points_a) != len(points_b):
        # enforce that align_a and align_b are the same length (they should be)
        raise ValueError('should not occur; bug in S-W local alignment?')
    if debug_print:
        print(points_a)
        print(points_b)
        print()
    # Create a nice neat form of this alignment
    # TODO add support for NP-only alignment gaps?
    range_a = [i for i in points_a if i >= 0]
    range_b = [i for i in points_b if i >= 0]
    range_a = (range_a[0], range_a[-1])
    range_b = (range_b[0], range_b[-1])
    output = pd.DataFrame(columns=[f'txt{i}' for i in range(
        (range_a[0] + range_b[0]) + len(points_a)
        + max(0, (len(align_a.columns) - range_a[1]) - 1)
        + max(0, (len(align_b.columns) - range_b[1]) - 1)
    )])
    # build the segment from align_a
    realign_a = align_a.loc[:, [f'txt{i}' for i in range(range_a[0])]]
    for i in range(range_b[0]):
        realign_a.insert(len(realign_a.columns), f'insx{i}', np.nan, True)
    for i in points_a:
        if i >= 0:
            realign_a[align_a.columns[i]] = align_a.loc[:, align_a.columns[i]]
        else:
            realign_a.insert(len(realign_a.columns), f'ins{len(realign_a.columns)}', np.nan, True)
    for i in range(range_a[1]+1, len(align_a.columns)):
        realign_a[align_a.columns[i]] = align_a.loc[:, align_a.columns[i]]
    for i in range(range_b[1]+1, len(align_b.columns)):
        realign_a.insert(len(realign_a.columns), f'insx{i+range_b[0]}', np.nan, True)
    # build the segment from align_b
    realign_b = align_b.loc[:, [f'txt{i}' for i in range(range_b[0])]]
    for i in range(range_a[0]):
        realign_b.insert(0, f'insx{i}', np.nan, True)
    for i in points_b:
        if i >= 0:
            realign_b[align_b.columns[i]] = align_b.loc[:, align_b.columns[i]]
        else:
            realign_b.insert(len(realign_b.columns), f'ins{len(realign_b.columns)}', np.nan, True)
    for i in range(range_a[1]+1, len(align_a.columns)):
        realign_b.insert(len(realign_b.columns), f'insx{i+range_a[0]}', np.nan, True)
    for i in range(range_b[1]+1, len(align_b.columns)):
        realign_b[align_b.columns[i]] = align_b.loc[:, align_b.columns[i]]
    # build final output
    realign_a.columns = output.columns
    realign_b.columns = output.columns
    output = output.append(realign_a)
    output = output.append(realign_b)
    return output.applymap(lambda x: ('', '', []) if x is np.nan else x), np.amax(scores, axis=None)


# Split the specified column into two columns, strictly stripping out the endmost token (either left-to-right or right-to-left)
# TODO-REFERENCE originally from alignment.ipynb
def splitSingleColumn(src_alignment, split_col, right_align=False):
    # do a quick sanity check to see if splitting would even make sense
    # there needs to be some column with more than one token that is even splittable
    if not all(src_alignment[split_col].map(lambda x: len(x[0].split(' '))>1)):
        return src_alignment
    # build the new split-col contents
    split = pd.DataFrame(columns=[f'{split_col}-0', f'{split_col}-1'])
    text_l = src_alignment[split_col].map(
        lambda x: ' '.join(x[0].split(' ')[:-1]) if right_align else x[0].split(' ')[0]
    )
    text_r = src_alignment[split_col].map(
        lambda x: x[0].split(' ')[-1] if right_align else ' '.join(x[0].split(' ')[1:])
    )
    # TODO ... use the parse tree to determine merged phrase POS?
    pos_l = src_alignment[split_col].map(
        lambda x: x[2][:-1] if right_align else [x[2][0]]
    )
    pos_r = src_alignment[split_col].map(
        lambda x: [x[2][-1]] if right_align else x[2][1:]
    )
    for index, row in src_alignment.iterrows():
        split.loc[index] = [
            (text_l[index], '', pos_l[index]),
            (text_r[index], '', pos_r[index]),
        ]
    output = src_alignment.copy()
    for i in range(len(split.columns)):
        output.insert(output.columns.get_loc(split_col), f'{split_col}-{i}', split[split.columns[i]])
    output = output.drop(split_col, 1)
    output.columns = [f'txt{i}' for i in range(len(output.columns))]
    return output

# Split the specified column into two columns using a word trie (either left-to-right or right-to-left)
# TODO-REFERENCE originally from alignment.ipynb
def splitTrieColumn(src_alignment, split_col, right_align=False):
    # side note, I use "tree" and "trie" interchangeably in these comments lol
    # define internal helper functions...
    # Column split step 1: Build word tree with node = word units running right->left
    # add text to the given trienode (this function edits the given tree_node)
    def wordTreeHelper(tree_node, text, id_data=None, right_align=False):
        text = text.strip()
        # Check for base case
        if text == '':
            tree_node[id_data] = id_data
            return tree_node
        # Select the right key
        key = ''
        if right_align:
            key = text.split(' ')[-1]
            text = ' '.join(text.split(' ')[0:-1])
        else:
            key = text.split(' ')[0]
            text = ' '.join(text.split(' ')[1:])
        # Put the key and text into the trie
        if key not in tree_node:
            tree_node[key] = {}
        tree_node[key] = wordTreeHelper(tree_node[key], text, id_data=id_data, right_align=right_align)
        return tree_node
    def wordTree(df, right_align=False):
        tree = {}
        for e_id in df.index:
            tree = wordTreeHelper(tree, df.loc[e_id], id_data=e_id, right_align=right_align)
        return tree
    # Column split step 2a: Collapse the suffix trie (merge nodes with only one child)
    # this function edits the input trie...
    def wordTreeCollapse(tree, right_align=False):
        # Collapse children nodes first
        added_keys = {}
        removed_keys = []
        for child in tree:
            # if child maps to more nodes
            if tree[child]!=child:
                tree[child] = wordTreeCollapse(tree[child], right_align=right_align)
                # Check if the new child node is collapsible
                if len(tree[child]) == 1:
                    merge_key = list(tree[child])[0]
                    if tree[child][merge_key]!=merge_key:
                        grandchild = list(tree[child])[0]
                        grandchild_tree = tree[child][grandchild]
                        # Perform the merge (put into edit queue)
                        removed_keys.append(child)
                        if right_align:
                            added_keys[grandchild + ' ' + child] = grandchild_tree
                        else:
                            added_keys[child + ' ' + grandchild] = grandchild_tree
        # Perform removals
        for key in removed_keys:
            tree.pop(key)
        # Perform additions
        for key in added_keys:
            tree[key] = added_keys[key]
        return tree
    # Column split step 2b: Compress the suffix trie to only two levels of depth.
    def wordTreeCompressHelper(tree_node, col_stack=[], right_align=False):
        updated_node = {}
        for k in tree_node:
            if k==tree_node[k]:
                # if this reaches a leaf node
                full_text = ' '.join(col_stack)
                if full_text not in updated_node:
                    updated_node[full_text] = {}
                updated_node[full_text][k]=k
            else:
                # if this is not a leaf node
                sub_node = wordTreeCompressHelper(
                    tree_node[k],
                    [k]+col_stack if right_align else col_stack+[k],
                    right_align=right_align
                )
                for sub_text in sub_node:
                    if sub_text not in updated_node:
                        updated_node[sub_text] = {}
                    for sub_value in sub_node[sub_text]:
                        updated_node[sub_text][sub_value] = sub_node[sub_text][sub_value]
        return updated_node
    # this function edits the input trie...
    def wordTreeCompress(tree, right_align=False):
        for root in tree:
            if root!=tree[root]:
                # compress the 2nd level onwards into a single level!
                compressed = wordTreeCompressHelper(tree[root])
                # remove the empty string artifact from compresshelper
                if '' in compressed:
                    for k in compressed['']:
                        compressed[k] = compressed[''][k]
                    compressed.pop('')
                tree[root] = compressed
        return tree
    # Column split step 3: Output the suffix trie to multiple columns
    # Calculate how many output columns we'll need
    # Get the depth of the trie (a trie with one terminal node {0:0} has depth 0)
    def wordTreeDepth(tree):
        max_depth = 0
        for child in tree:
            if tree[child]!=child:
                max_depth = max(max_depth, 1 + wordTreeDepth(tree[child]))
        return max_depth
    def wordTreeSplit(tree, max_depth, output, so_far=[], right_align=False):
        for child in tree:
            if tree[child]==child:
                # we have hit a base, put in an entry
                if right_align:
                    output[child] = ['']*(max_depth - len(so_far)) + so_far
                else:
                    output[child] = so_far + ['']*(max_depth - len(so_far))
            else:
                # this node has further children!
                output = wordTreeSplit(
                    tree[child],
                    max_depth,
                    output,
                    ([child] + so_far) if right_align else (so_far + [child]),
                    right_align=right_align)
        return output
    # Now actually do the split operation
    tree = wordTree(src_alignment[split_col].map(lambda x: x[0]), right_align=right_align)
    tree = wordTreeCollapse(tree, right_align=right_align)
    # compress that tree into only 2 levels because we only want to generate ONE new column max
    tree = wordTreeCompress(tree, right_align=right_align)
    tree_depth = wordTreeDepth(tree)
    split_data = wordTreeSplit(tree, tree_depth, {}, right_align=right_align)
    # TODO ... use the parse tree to determine merged phrase POS?
    pos_info = src_alignment[split_col].map(lambda x: x[2])
    for k in split_data:
        pos_i = 0
        for i in range(len(split_data[k])):
            chunk_text = split_data[k][i]
            chunk_len = 0 if len(chunk_text)==0 else len(chunk_text.split(' '))
            split_data[k][i] = (chunk_text, '', pos_info[k][pos_i:pos_i+chunk_len])
            pos_i += chunk_len
    split = pd.DataFrame(columns=[f'{split_col}-{i}' for i in range(tree_depth)])
    for id in split_data:
        split.loc[id] = split_data[id]
    # combine the split results back into the rest of our original input data!
    output = src_alignment.copy()
    for i in range(len(split.columns)):
        output.insert(output.columns.get_loc(split_col), f'{split_col}-{i}', split[split.columns[i]])
    output = output.drop(split_col, 1)
    output.columns = [f'txt{i}' for i in range(len(output.columns))]
    return output


# Merge the specified column in an alignment with its immediate neighbor to the right.
# Does nothing if the given column is already to the rightmost
# TODO-REFERENCE originally from alignment.ipynb
def mergeColumn(src_alignment, merge_col):
    merge_col_next = src_alignment.columns[list(src_alignment.columns).index(merge_col)+1]
    # combine the segment (text)
    merged_segment = src_alignment[merge_col].map(lambda x: x[0]) + ' ' + src_alignment[merge_col_next].map(lambda x: x[0])
    merged_segment = merged_segment.map(lambda x: x.strip())
    # combine the pos (phrase pos)
    merged_pos = src_alignment[merge_col].map(lambda x: x[1]) + ' ' + src_alignment[merge_col_next].map(lambda x: x[1])
    merged_pos = merged_pos.map(lambda x: x.strip().split())
    # TODO ... use the parse tree to determine merged phrase POS?
    merged_pos = merged_pos.map(lambda x: '' if len(x)==0 else x[0])
    # combine the cpos (token / word pos)
    merged_cpos = src_alignment[merge_col].map(lambda x: x[2]) + src_alignment[merge_col_next].map(lambda x: x[2])
    # put the result column into our result
    result = src_alignment.copy()
    result[merge_col] = list(zip(merged_segment, merged_pos, merged_cpos))
    del result[merge_col_next]
    result.columns = [f'txt{i}' for i in range(len(result.columns))]
    return result


# Return true iff the specified shift can occur without causing text collisions
# TODO-REFERENCE originally from alignment.ipynb
def canShiftCells(src_alignment, shift_rows, shift_col, shift_distance, shift_size):
    # remove duplicates
    shift_rows = list(set(shift_rows))
    # check that the selected segment starting point(s) exist
    if not all([(e in src_alignment.index) for e in shift_rows]):
        return False
    if shift_col not in src_alignment.columns:
        return False
    # get the index numbers we are working with
    colindex_start = list(src_alignment.columns).index(shift_col)
    # check that the entire selected segment is contained within the alignment
    if colindex_start + shift_size > len(src_alignment.columns):
        return False
    # check that the proposed shift is contained within the alignment
    if (colindex_start + shift_distance) < 0 or (colindex_start + shift_distance) >= len(src_alignment.columns):
        return False
    if (colindex_start + (shift_size-1) + shift_distance) < 0 or (colindex_start + (shift_size-1) + shift_distance) >= len(src_alignment.columns):
        return False
    # check that the alignment segment(s) is entirely text and does not contain whitespace
    if any([any([len(e[0].strip())==0 for e in src_alignment.loc[shift_row][colindex_start:colindex_start+shift_size]]) for shift_row in shift_rows]):
        return False
    # if the shift distance is 0, it always works (although it's a very useless shift)
    if shift_distance==0:
        return True
    # figure out if the shift collides with any other text for each of the rows we want to shift
    for shift_row in shift_rows:
        if shift_distance > 0:
            can_reach = [
                i for i
                in range(colindex_start+shift_size, min(len(src_alignment.loc[shift_row]), colindex_start+shift_size+shift_distance))
            ]
        elif shift_distance < 0:
            can_reach = [
                i for i
                in reversed(range(max(0, colindex_start+shift_distance), colindex_start))
            ]
        can_reach = [(i, src_alignment.loc[shift_row][i][0].strip()=='') for i in can_reach]
        # check whether we should continue with the shift
        if not all([e[1] for e in can_reach]):
            return False
    return True


# Shifts the specified cells in an alignment
# If it is impossible to shift the cells as specified, throws a ValueError
# TODO-REFERENCE originally from alignment.ipynb
def shiftCells(src_alignment, shift_rows, shift_col, shift_distance, shift_size=1, force_push=False, emptycell=('','',[]), debug_print=False):
    if debug_print:
        print(f'shift rows {shift_rows}, {shift_size} cells starting from {shift_col}, {shift_distance} cells over')
    # check if it's possible to shift
    if not force_push and not canShiftCells(src_alignment, shift_rows, shift_col, shift_distance, shift_size):
        raise ValueError(
            'impossible to shift with given parameters: '
            + f'(shift row {shift_rows}, {shift_size} cells starting from {shift_col}, {shift_distance} cells over)'
        )
    # initialize the alignment table copy we'll be working with
    result = src_alignment.copy()
    # get the index numbers we are working with
    colindex_start = list(result.columns).index(shift_col)
    for shift_row in shift_rows:
        row_colindex_start = colindex_start
        row_shift_size = shift_size
        if force_push:
            # if force_push is True, then expand the selection to include stuff that gets pushed
            shift_direction = int(shift_distance/abs(shift_distance))
            next_cell_text = 'temp_init'
            while len(next_cell_text)>0:
                if (shift_direction<0) and (row_colindex_start>0):
                    next_cell_text = result.loc[shift_row][row_colindex_start+shift_direction][0].strip()
                elif (shift_direction>0) and (row_colindex_start+row_shift_size<len(result.columns)):
                    next_cell_text = result.loc[shift_row][row_colindex_start+row_shift_size+shift_direction-1][0].strip()
                else:
                    next_cell_text = ''
                if len(next_cell_text)>0:
                    if shift_direction<0:
                        row_colindex_start -= 1
                    row_shift_size += 1
            # check if it's possible to push all of this
            if not canShiftCells(src_alignment, [shift_row], result.columns[row_colindex_start], shift_distance, row_shift_size):
                raise ValueError(
                    'impossible to shift with given parameters (even with force_push=True): '
                    + f'(shift row {shift_rows}, {shift_size} cells starting from {shift_col}, {shift_distance} cells over)'
                )
        # grab the old contents
        clipboard = [e for e in result.loc[shift_row][row_colindex_start:row_colindex_start+row_shift_size]]
        # replace old contents with empty tuples
        for i in range(row_colindex_start, row_colindex_start+row_shift_size):
            result.loc[shift_row][i] = emptycell
        # put old content in its destination location
        for i in range(len(clipboard)):
            result.loc[shift_row][row_colindex_start+shift_distance+i] = clipboard[i]
    return result # removeEmptyColumns(result)


# Squishes (merges) the specified single cell in an alignment with its neighbor
# shift_direction=1 means shift to the right, shift_direction=-1 means shift to the left
def squishCells(src_alignment, shift_row, shift_col, shift_direction, emptycell=('','',[])):
    # initialize the alignment table copy we'll be working with
    result = src_alignment.copy()
    # get the index numbers we are working with
    colindex_start = list(result.columns).index(shift_col)
    # grab the source cell
    source = result.loc[shift_row][colindex_start]
    # replace old cell with empty tuple
    result.loc[shift_row][colindex_start] = emptycell
    # merge old content with destination content
    dest = result.loc[shift_row][colindex_start+shift_direction]
    # TODO implement phrase POS merging somehow
    if shift_direction > 0:
        merged = (source[0]+' '+dest[0], '', source[2]+dest[2])
    else:
        merged = (dest[0]+' '+source[0], '', dest[2]+source[2])
    result.loc[shift_row][colindex_start+shift_direction] = merged
    return result


# Insert a column into an alignment relative (before or after) to the specified column
def insertColumn(src_alignment, insert_col, insert_after=True, emptycell=('','',[])):
    # initialize the alignment table copy we'll be working with
    result = src_alignment.copy()
    # get the column index number we are working with
    colindex = list(result.columns).index(insert_col)
    # calculate column index based on whether we are inserting before or after
    if insert_after:
        colindex = colindex + 1
    # perform the actual insert
    result.insert(loc=colindex, column='newcol', value=[emptycell for i in range(len(result))])
    # rename all of the columns...
    result.columns = [f'txt{i}' for i in range(len(result.columns))]
    return result


# Delete the specified column from the alignment ... even if it has contents
def deleteColumn(src_alignment, delete_col):
    # initialize the alignment table copy we'll be working with
    result = src_alignment.copy()
    # perform the actual deletion
    result = result.drop(columns=[delete_col])
    # rename all of the columns...
    result.columns = [f'txt{i}' for i in range(len(result.columns))]
    return result


# calculate total column count
# TODO-REFERENCE originally from alignment.ipynb
def scoreNumColumns(align_df):
    return len(align_df.columns)


# calculate how much we should care about different terms in the alignment
# TODO-REFERENCE originally from alignment.ipynb
def alignmentTermWeights(align_df, sp, all_stopwords=None, priority_pos=['NN', 'NNS', 'NNP', 'JJ', 'RB']):
    # input argument sp: spacy model (default should be sp = spacy.load('en_core_web_sm'))
    if all_stopwords is None:
        all_stopwords = sp.Defaults.stop_words
    # collect list forms of words and cPOS
    all_text = [
        [text for text in extractTup(align_df.iloc[rownum], tup_i='segment', is_frame=False)]
        for rownum in range(len(align_df))
    ]
    all_text = [' '.join(sublist).split() for sublist in all_text]
    all_cpos = [
        [text for sublist
         in extractTup(align_df.iloc[rownum], tup_i='cpos', is_frame=False)
         for text in sublist]
        for rownum in range(len(align_df))
    ]
    # get count of how many rows each word is present in
    tokens_df = dict([
        (word, sum([(word in row) for row in all_text]))
        for word
        in set([item for sublist in all_text for item in sublist])
        if word not in all_stopwords
    ])
    # remove the words that show up in at most one row
    for word in [word for word in tokens_df if tokens_df[word] <= 1]:
        discard = tokens_df.pop(word, None)
    # flatten the word and cPOS lists
    all_text = [e for sublist in all_text for e in sublist]
    all_cpos = [e for sublist in all_cpos for e in sublist if e != '']
    # count up how many POS is assigned to each word
    pos_mapping = {}
    for i in range(len(all_text)):
        if all_text[i] not in pos_mapping:
            pos_mapping[all_text[i]] = {}
        if all_cpos[i] not in pos_mapping[all_text[i]]:
            pos_mapping[all_text[i]][all_cpos[i]] = 0
        pos_mapping[all_text[i]][all_cpos[i]] += 1
    # pick the single POS that each word is tagged as most often
    for word in pos_mapping:
        max_pos = None
        max_count = 0
        for pos in pos_mapping[word]:
            if pos_mapping[word][pos] > max_count:
                max_pos = pos
                max_count = pos_mapping[word][pos]
        pos_mapping[word] = max_pos
    # exponentiate the count of all of the words in the dict that are in POS classes we care about
    for word in tokens_df:
        if any([(pos in pos_mapping[word]) for pos in priority_pos]):
            tokens_df[word] = pow(tokens_df[word], 2)
    return tokens_df


# calculate a subscore for how many columns are represented in the alignment
# TODO-REFERENCE originally from alignment.ipynb
def scoreColumnRepresentation(align_df, colname):
    # Count the fraction of rows that are represented in the column (so penalizes gaps)
    tokens = [text for text in extractTup(align_df[colname], tup_i='segment', is_frame=False)]
    non_empty_count = len([text for text in tokens if text.strip() != ''])
    return non_empty_count/len(tokens)


# calculate a subscore for how many tokens there are in a column in the alignment
# TODO-REFERENCE originally from alignment.ipynb
def scoreColumnTotalTokens(align_df, colname):
    # Count the number of words (including repeats) in each column
    tokens = [text.split(' ') for text in extractTup(align_df[colname], tup_i='segment', is_frame=False)]
    tokens = [e for sublist in tokens for e in sublist if e!='']
    return len(tokens)


# calculate a subscore for how many columns are filled in the alignment
# TODO-REFERENCE originally from alignment.ipynb
def scoreNumFilledColumns(align_df):
    # empty columns don't count, so discount those...
    contents = [align_df[colname] for colname in align_df.columns]
    # count how many texts are in each column
    contents = [len([1 for cell in t if cell[0].strip()!='']) for t in contents]
    # count how many columns have some text in them
    contents = [(1 if t>0 else 0) for t in contents]
    return sum(contents)


# calculate a subscore for the word vector embed variance per column in an alignment
# TODO-REFERENCE originally from alignment.ipynb
def scoreColumnPhraseEmbedVariance(align_df, colname, embed_model):
    # Compute embeddings variance of all the phrases for a single column
    texts = [text for text in extractTup(align_df[colname], tup_i='segment', is_frame=False)]
    # take the set of row texts
    texts = list(set(texts))
    # build up the list of text embeds for the texts that we *can* compute an embed for
    text_embeds = []
    for word in texts:
        try:
            text_embeds.append(get_phrase_embed(embed_model, word).drop('word', 1))
        except:
            pass
    if len(text_embeds) > 1:
        output = pd.concat(text_embeds)
        # Reasoning for this operation (calculating variance as trace(covariance matrix) ...):
        # https://stats.stackexchange.com/questions/225434/a-measure-of-variance-from-the-covariance-matrix
        # This is equivalent to calculating the expected Euclidean distance of each element from the mean
        result = np.trace(output.cov())
    else:
        # one of two scenarios:
        # 1. all of the contents of this column aren't considered words, so, pretend they're all the same
        # 2. there is only one row in this column that contains text, so it has no variation
        # TODO is there a theoretically better way to handle them?
        result = 0
    return result


# calculate a subscore for how many unique tokens there are per column in an alignment
# TODO-REFERENCE originally from alignment.ipynb
def scoreColumnTokenCount(align_df, colname):
    # TODO normalize this
    # Count the number of unique tokens in a single column
    # capture each cell text
    tokens = [
        text for text in extractTup(align_df[colname], tup_i='segment', is_frame=False)
        if text.strip() != ''
    ]
    # split it into tokens
    tokens = [text.split() for text in tokens]
    # flatten
    return len(set([token for sublist in tokens for token in sublist]))


# calculate a subscore for how many unique entities there are per column in an alignment
# TODO-REFERENCE originally from alignment.ipynb
def scoreColumnTokenEntityCount(align_df, colname, scisp, scisp_linker):
    # input argument scisp: scispacy model (default should be scisp = spacy.load('en_core_sci_sm'))
    # input argument scisp_linker: scispacy model linker
    # Count the number of unique entity types in a single column
    types = [text for text in extractTup(align_df[colname], tup_i='segment', is_frame=False)
             if text.strip() != '']
    # process the texts through spacy and pick out entities
    types = [scisp(text).ents for text in types]
    # flatten the entities and put into a set
    types = [[ent for ent in ents] for ents in types]
    # get the UMLS mappings for each entity
    types = [[ent._.umls_ents[0][0] for ent in sl if (len(ent)>0) and (len(ent._.umls_ents)>0)] for sl in types]
    # get the TUI for each of these UMLS mappings
    # An informal guide to all of the TUIs: https://gist.github.com/joelkuiper/4869d148333f279c2b2e
    types_tui = [[scisp_linker.umls.cui_to_entity[ent].types for ent in sl] for sl in types]
    # check for edge case where there's 0 UMLS tuis for something?
    for sl in types_tui:
        if any([len(e)<1 for e in sl]):
            raise ValueError('<1 tui for a UMLS entity')
    types_tui = [[e for sl in row for e in sl] for row in types_tui]
    # TODO implement larger groupings of types / more general type groups...
    # https://semanticnetwork.nlm.nih.gov/download/SemGroups.txt
    # from https://semanticnetwork.nlm.nih.gov/
    return len(set([e for sl in types for e in sl])), len(set([e for sl in types_tui for e in sl]))


# calculate a subscore for how many columns a given term appears in in the alignment
# TODO-REFERENCE originally from alignment.ipynb
def scoreTermColumnCount(align_df, term):
    # Count the number of columns that a certain phrase or term appears within
    # TODO should this be a fraction instead? what would that imply?
    # If it doesn't appear at all, returns 1 (TODO that might not be ideal?)
    # TODO add support for regex patterns (eg numbers?)
    tokens = [
        [text for text in extractTup(align_df[colname], tup_i='segment', is_frame=False)]
        for colname in align_df.columns
    ]
    tokens = [[e for e in col if (term.lower() in e.lower())] for col in tokens]
    tokens = [col for col in tokens if len(col) != 0]
    return max(1, len(tokens))


# calculate a subscore for how many columns a given list of terms appear in in the alignment
# TODO-REFERENCE originally from alignment.ipynb
def scoreTermListColumnCount(align_df, term_list, term_weights=None):
    # if we don't have any terms to investigate, return 1 (default col count)
    if len(term_list) == 0:
        return 1
    # by default, weight each term equally
    if term_weights is None:
        term_weights = [1]*len(term_list)
    # And normalize the weights (assume that hasn't been done already)
    tw_sum = sum(term_weights)
    term_weights = [(tw/tw_sum) for tw in term_weights]
    scores = [scoreTermColumnCount(align_df, term) for term in term_list]
    return np.dot(scores, term_weights)


# calculate the maximum row length of an alignment
# (the number of filled cells in the row with the most filled cells)
def maxRowLength(align_df):
    return max([len([1 for e in align_df.loc[i] if len(e[0].strip())!=0]) for i in align_df.index])


# calculate an alignment overall score
# TODO-REFERENCE originally from alignment.ipynb
def scoreAlignment(align_df, spacy_model, scispacy_model, scispacy_linker, embed_model, term_weight_func=None, weight_components=None):
    # set default score weights...
    if weight_components is None:
        print('scoreAlignment: using default weight_components')
        weight_components = np.array([0.2, 0.2, 1, 0, 0, 0])
    # calculate the max_row_length field...
    max_row_length = maxRowLength(align_df)

    # GET SCORE COMPONENTS
    # calculate score_numcolumns subscore
    if not weight_components[0]==0 or True:
        score_numcolumns = scoreNumColumns(align_df)
    else:
        score_numcolumns = 0
    # calculate score_numfilledcolumns subscore
    if not weight_components[1]==0 or True:
        score_numfilledcolumns = scoreNumFilledColumns(align_df)
    else:
        score_numfilledcolumns = 0
    # calculate score_colptxtembed subscore
    if not weight_components[2]==0 or True:
        score_colptxtembed = [scoreColumnPhraseEmbedVariance(align_df, colname, embed_model) for colname in align_df.columns]
    else:
        score_colptxtembed = [0 for colname in align_df.columns]
    # calculate score_coltokncount subscore
    if not weight_components[3]==0:
        score_coltokncount = [scoreColumnTokenCount(align_df, colname) for colname in align_df.columns]
    else:
        score_coltokncount = [0 for colname in align_df.columns]
    # calculate raw_colentityscores subscore
    if not weight_components[4]==0:
        raw_colentityscores = [
            scoreColumnTokenEntityCount(align_df, colname, scisp=scispacy_model, scisp_linker=scispacy_linker)
            for colname in align_df.columns
        ]
    else:
        raw_colentityscores = [(0,0) for colname in align_df.columns]
    score_coltentcount = [s[0] for s in raw_colentityscores]
    score_colttuicount = [s[1] for s in raw_colentityscores]
    # calculate score_termcolcount subscore
    if not weight_components[5]==0:
        # get term weights...
        if term_weight_func is None:
            alignment_terms = alignmentTermWeights(align_df, sp=spacy_model)
            term_list = list(alignment_terms)
            weight_terms = list(alignment_terms.values())
        weight_terms = [r/sum(weight_terms) for r in weight_terms] # normalize the weights
        score_termcolcount = scoreTermListColumnCount(align_df, term_list, weight_terms)
    else:
        score_termcolcount = 0
    # get column weights...
    score_colalltokens = [scoreColumnRepresentation(align_df, colname) for colname in align_df.columns]
    #     score_colalltokens = [scoreColumnTotalTokens(align_df, colname) for colname in align_df.columns]
    #     score_colalltokens = [1 for colname in align_df.columns]
    weight_columns = [r/sum(score_colalltokens) for r in score_colalltokens] # normalize the column weights

    # put score components into a df of their own that is neatly readable for debug purposes
    rawscores = pd.DataFrame([
        # text embed vector variance; lower is better
        score_colptxtembed,
        # distinct tokens; lower is better
        score_coltokncount,
        # varying distinct entity TUIs; lower is better
        score_colttuicount,
        # the weighting we are giving each column; higher means more attention
        weight_columns,
    ], index=[
        'embed variance',
        'unique tokens (var)',
        'unique entity TUI (var)',
        'relevance (numtokens)',
    ])
    rawscores = rawscores.rename(columns=dict(zip(rawscores.columns, align_df.columns)))

    # apply column weights to the score components
    components = [
        # number of columns; lower is better
        -1 * math.pow((score_numcolumns / max_row_length), 1),
        # number of filled columns; lower is better
        -1 * math.pow((score_numfilledcolumns / max_row_length), 1),
        # text embed vector variance; lower is better
        -1 * np.dot([math.pow(s, 2) for s in score_colptxtembed], weight_columns),
        # distinct tokens; lower is better
        -1 * np.dot(score_coltokncount, weight_columns),
        # distinct entity TUIs; lower is better
        -1 * np.dot(score_colttuicount, weight_columns),
        # column count of terms used; lower is better
        -1 * score_termcolcount,
    ]
    components = np.array(components)
    # apply score component weights (higher total score is better)
    bias = 5
    singlescore = bias + np.dot(weight_components, components)
    return singlescore, components, rawscores
