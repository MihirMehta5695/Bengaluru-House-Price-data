from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/')
def hello():
    return "Hi"


if __name__ == "__main__":
    print('Starting Python Flask server for Home Price Prediction')
    app.run()
