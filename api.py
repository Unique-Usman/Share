#!/usr/bin/python3
"""
Deals with the RESTful API action(s) of the general page. It will soon change though.
"""

from api.v1.views import app_views
from flask import jsonify, request
#from models.model import qa_bot
from models.research.test import create_app as qa_bot

supervaani_chains = {}

@app_views.route("/<userID>/supervaani", methods=['POST'], strict_slashes=False)
def handle_supervaani(userID):
    content_type = request.headers.get("Content-Type")
    if content_type != "application/json":
        return jsonify({"message": "Not a JSON"}), 400
    content = request.get_json()
    user_input = content.get("user_message", None)
    if user_input is None:
        return jsonify({"message": "Missing user_message"}), 400
    supervaani_chain = supervaani_chains.get(userID, None)
    if supervaani_chain is None:
        supervaani_chain = qa_bot()
        supervaani_chains[userID] = supervaani_chain
    try:
        result = supervaani_chain.invoke({"question": user_input})
        result = result.get("generation", None)
    except Exception as e:
        result = "Sorry, something went wrong. contact the developer"
    return jsonify({"supervaani_message": result}), 200

@app_views.route("/<string:userID>/leave", methods=['POST'], strict_slashes=False)
def handle_leave(userID):
    if userID in supervaani_chains:
        del supervaani_chains[userID]
    return jsonify({"supervaani_message": "User left"}), 200
