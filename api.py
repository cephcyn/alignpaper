from flask import Flask
from flask import request
import traceback
import json
from os import path

# Flask-specific code...
app = Flask(__name__)

@app.route('/api', methods=['GET'])
def api():
	# retrieve arguments
	try:
		# arg_id = int(request.args['id'])
		arg_data = request.args['word'].split('\n') if ('word' in request.args) else ['default']
	except:
		return {'error':'improperly formatted or missing arguments','traceback':f'{traceback.format_exc()}'}
	print('arg_data:', arg_data)
	if path.isfile(f'testcases/{arg_data[0]}/a.json'):
		# TODO temporary - read the temp data file
		print('=== READING FILE FROM DISK AS IT EXISTS!!!: ===')
		print(f'testcases/{arg_data[0]}/a.json')
		with open(f'testcases/{arg_data[0]}/a.json') as f:
			data = json.load(f)
	else:
		print('=== WRITING DATA DIRECTLY IN AS ALIGNMENT DATA: ===')
		data = {
			'alignment': [
				{
					'id': i,
					'pos': [['TMP']],
					'txt': [[arg_data[i]]],
				} for i in range(len(arg_data))
			]
		}
		print(data)
	data['temp_arg_data'] = arg_data
	# build output
	return data
