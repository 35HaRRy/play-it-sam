import os

import yaml
# import tiktoken
import spotipy.util as util

from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.agent_toolkits.openapi import planner

from langchain_community.utilities.requests import RequestsWrapper

from langchain_groq import ChatGroq

# with open("spotify_openapi.yaml") as f:
#     raw_spotify_api_spec = yaml.load(f, Loader=yaml.Loader)

with open("spotify_openapi.yaml", "r", encoding="utf-8") as file:
    raw_spotify_api_spec = yaml.safe_load(file)
spotify_api_spec = reduce_openapi_spec(raw_spotify_api_spec)


def construct_spotify_auth_headers(raw_spec: dict):
    scopes = list(
        raw_spec["components"]["securitySchemes"]["oauth_2_0"]["flows"][
            "authorizationCode"
        ]["scopes"].keys()
    )
    access_token = util.prompt_for_user_token(scope=",".join(scopes))
    return {"Authorization": f"Bearer {access_token}"}


# Get API credentials.
headers = construct_spotify_auth_headers(raw_spotify_api_spec)
requests_wrapper = RequestsWrapper(headers=headers)

# llm = ChatGroq(model_name="gemma2-9b-it", temperature=0.0)
llm = ChatGroq(model_name="llama-3.1-70b-versatile", temperature=0.0)

# NOTE: set allow_dangerous_requests manually for security concern 
# https://python.langchain.com/docs/security
spotify_agent = planner.create_openapi_agent(
    spotify_api_spec,
    requests_wrapper,
    llm,
    allow_dangerous_requests=True,
)
# user_query = (
#     "make me a playlist with the first song from kind of blue."
#     "call it machine blues."
# )
user_query = (
    "'Test' çalma listemdeki ilk 5 şarkıyı kategorize edip listeler misin?"
)
spotify_agent.invoke(user_query)

# endpoints = [
#     (route, operation)
#     for route, operations in raw_spotify_api_spec["paths"].items()
#     for operation in operations
#     if operation in ["get", "post"]
# ]
# len(endpoints)

# enc = tiktoken.encoding_for_model("gpt-4")

# def count_tokens(s):
#     return len(enc.encode(s))

# count_tokens(yaml.dump(raw_spotify_api_spec))
