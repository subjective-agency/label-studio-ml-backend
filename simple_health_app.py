#!/usr/bin/env python3
"""
Simple health check endpoint for Label Studio
"""

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health', methods=['GET'])
@app.route('/', methods=['GET'])
def health():
    return jsonify({
        'status': 'UP',
        'model_class': 'BaseModel'
    })

if __name__ == "__main__":
    print("Starting health check server on 0.0.0.0:9090")
    app.run(host='0.0.0.0', port=9090, debug=True) 