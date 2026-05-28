from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return '<h1>Hello from my CI/CD powered Flaskapp!</h1>'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
