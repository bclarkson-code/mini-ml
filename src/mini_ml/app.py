"""
A server that allows users to train models and get predictions from them.
"""
from flask import Flask, request

app = Flask(__name__)


@app.route("/train")
def train() -> str:
    """
    Given some data, train a model and
    return the cross validation score
    """
    print(request)
    return "Hello World"


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
