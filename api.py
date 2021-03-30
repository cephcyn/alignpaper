from flask import Flask
from flask import request
import traceback

app = Flask(__name__)

@app.route('/api', methods=['GET'])
def api():
	# retrieve arguments
	try:
		id = int(request.args['id'])
		if 'word' in request.args:
			word = request.args['word']
		else:
			word = 'default'
	except:
		return {'error':'improperly formatted or missing arguments','traceback':f'{traceback.format_exc()}'}
	# build output
	return {'id':id,'word':word,'somearray':['foo','bar','baz']}


