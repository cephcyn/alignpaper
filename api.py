from flask import Flask
from flask import request
import traceback
import json
from os import path
import pandas as pd
import numpy as np

import alignment

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
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
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
import gensim
# Load fasttext-wiki-news-subwords-300 pretrained model
fasttext = gensim.models.keyedvectors.FastTextKeyedVectors.load('model/fasttext-wiki-news-subwords-300.model', mmap='r')

# # TODO-REFERENCE originally from alignment.ipynb
# import spacy
# sp = spacy.load('en_core_web_sm')
# import scispacy
# from scispacy.linking import EntityLinker
# scisp = spacy.load('en_core_sci_sm')
# linker = scisp.add_pipe('scispacy_linker', config={'resolve_abbreviations': True, 'linker_name': 'umls'})

print('=== FINISHED NLP MODEL IMPORTS ===')

# Flask-specific code...
app = Flask(__name__)

@app.route('/api', methods=['GET'])
def api():
	# retrieve arguments
	try:
		# arg_id = int(request.args['id'])
		arg_input = request.args['input'].split('\n') if ('input' in request.args) else ['default']
	except:
		return {'error':'improperly formatted or missing arguments','traceback':f'{traceback.format_exc()}'}
	print('arg_input:', arg_input)
	if path.isfile(f'testcases/{arg_input[0]}/a.json'):
		# TODO temporary - read the temp data file
		print('=== READING FILE FROM DISK AS IT EXISTS!!!: ===')
		print(f'testcases/{arg_input[0]}/a.json')
		with open(f'testcases/{arg_input[0]}/a.json') as f:
			data = json.load(f)
	else:
		print('=== WRITING DATA DIRECTLY IN AS ALIGNMENT DATA: ===')
		data = {}
		# retrieve the constituency parse information
		data['parse_constituency'] = dict(zip(
			range(len(arg_input)),
			[alignment.parse_constituency(constituency_predictor, p) for p in arg_input]
		))
		print(data['parse_constituency'])
		print('===')
		# build the raw input df that the alignment and search algorithms build on top of...
		input_df_dict = {}
		for txt_id in data['parse_constituency']:
			tokens = []
			for token_i in range(len(data['parse_constituency'][txt_id]['tokens'])):
				tokens.append(
					(
						data['parse_constituency'][txt_id]['tokens'][token_i],
						'',
						[data['parse_constituency'][txt_id]['pos_tags'][token_i]],
					)
				)
			input_df_dict[txt_id] = tokens
		input_df = pd.DataFrame(input_df_dict.values(), index=input_df_dict.keys())
		input_df = input_df.applymap(lambda x: ('', '', []) if (x is None) else x)
		input_df.columns = [f'txt{i}' for i in range(len(input_df.columns))]
		# data['input_df'] = input_df
		print(input_df)
		print('===')
		# align the texts!
		# TODO adapt this to multiple input sizes
		align_df, align_df_score = alignment.alignRowMajorLocal(
			input_df.loc[[0]],
			input_df.loc[[1]],
			embed_model=fasttext
		)
		print(align_df)
		print('===')
		# convert the final alignment output to an outputtable format
		data['alignment'] = alignment.alignment_to_jsondict(align_df)['alignment']
		print(data['alignment'])
		print('===')
	data['temp_arg_input'] = arg_input
	# build output
	return data
