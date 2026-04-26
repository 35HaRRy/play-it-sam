# play-it-sam

## Overview
A LangChain-based Spotify assistant that plans actions with an OpenAPI workflow and executes them against the Spotify API.
It is designed as an experimental AI project for account and playlist management.

## Dependencies
- Python
- `langchain-core`
- `langchain-community`
- `langgraph`
- `langchain-groq`
- `spotipy`
- `PyYAML`
- `python-dotenv`
- `requests`

## Setup
1. Create and activate a virtual environment.
2. Install the Python dependencies.
3. Add Spotify credentials and any model-provider keys to your `.env` file.
4. Review `spotify_openapi.yaml` if you want to customize the API planning flow.

## Run
- Launch the interactive assistant with `python start.py`.
- Use the included notebooks for experimentation or prompt planning.

## Notes
The project is still experimental, so you may need to adjust prompts, credentials, or API scopes depending on your Spotify account and workflow.