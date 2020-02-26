import sys
import logging

from flask import Flask, jsonify, request

from solution.settings import baseline_items
from solution.get_recs import predict

app = Flask(__name__)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s")


@app.route('/ready')
def ready():
    return "OK"


@app.route('/recommend', methods=['POST'])
def recommend():
    req = request.json
    try:
        items_ranked = predict(req)
        # uncomment for debug
        # logging.info('OK, cnt_items: %s', str(len(items_ranked)))
    except Exception as e:
        items_ranked = baseline_items.copy()
        # logging.info('ERROR, cnt_items: %s', str(len(items_ranked)))
        logging.exception(e)

    return jsonify({
        'recommended_products': items_ranked
    })


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=8000)
