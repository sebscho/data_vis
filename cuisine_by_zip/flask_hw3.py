from flask import Flask, Response
from cuisineChart import showCuisines, loadData

data = loadData()
app = Flask(__name__, static_url_path='', static_folder='.')
app.add_url_rule('/', 'root', lambda: app.send_static_file('HW3index.html'))


#@app.route('/vis/int:<zipCode>')
@app.route('/vis/<zipCode>')
def makeVis(zipCode):

    output = showCuisines(data, zipCode).to_json()

    return Response(output,
        mimetype='application/json',
        headers={
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*'
            }
        )

if __name__ == '__main__':
    app.run(port=8002)
