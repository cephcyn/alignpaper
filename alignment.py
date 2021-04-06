import warnings
import copy

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
