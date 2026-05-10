from flask import Flask, render_template, request


app = Flask(__name__)

@app.route('/')
def hello_world():
    return '<h1>Welcome to Flask!</h1>'

@app.route('/welcome')
def welcome():
    return '<h2>Welcome navigation</h2>'

@app.route('/welcome/<name>')
def welcome_name(name):
    return f'<h2>Welcome, {name}!</h2>'

@app.route('/square', methods=['GET'])
def squarenumber():
    if request.method == 'GET':
        if(request.args.get('num') == None): # when user requests first time, it will be None
            return render_template('/square.html') # by default Flask looks in the templates folder for the html file
        elif(request.args.get('num') == ''):
            return "<html><body> <h1>Invalid input</h1></body></html>"
        else:
            number = request.args.get('num')
            sqare = int(number) * int(number)
            return render_template('/solution.html',
                                   squareofnum=sqare, num=number)


if __name__ == '__main__':
    app.run(debug=True)

