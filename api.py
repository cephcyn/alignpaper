from flask import Flask
from flask import request
import traceback
import json
from os import path

app = Flask(__name__)

@app.route('/api', methods=['GET'])
def api():
	# retrieve arguments
	try:
		# arg_id = int(request.args['id'])
		if 'word' in request.args:
			arg_data = request.args['word'].split('\n')
		# else:
		# 	arg_data = ['default']
	except:
		return {'error':'improperly formatted or missing arguments','traceback':f'{traceback.format_exc()}'}
	print('arg_data:', arg_data)
	# TODO temporary - read the temp data file
	if path.isfile(f'testcases/{arg_data[0]}/a.json'):
		print('Reading file from disk as it exists!!!:')
		print(f'testcases/{arg_data[0]}/a.json')
		with open(f'testcases/{arg_data[0]}/a.json') as f:
			data = json.load(f)
	else:
		with open(f'temp/testjsonformatalignment.json') as f:
			data = json.load(f)
	data['temp_arg_data'] = arg_data
	# build output
	return data
