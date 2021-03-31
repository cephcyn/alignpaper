from flask import Flask
from flask import request
import traceback
import json

app = Flask(__name__)

@app.route('/api', methods=['GET'])
def api():
	# retrieve arguments
	try:
		arg_id = int(request.args['id'])
		if 'word' in request.args:
			arg_word = request.args['word']
		else:
			arg_word = 'default'
	except:
		return {'error':'improperly formatted or missing arguments','traceback':f'{traceback.format_exc()}'}
	# TODO temporary - read the temp data file
	with open(f'temp/testjsonformatalignment.json') as f:
		data = json.load(f)
	data['argument_data'] = f'{arg_id}_{arg_word}'
	# build output
	return data
