import os
import yaml
import dotenv
import asyncio
import operator

import spotipy.util as util

from typing import Annotated, List, Tuple, TypedDict, Union, Literal

from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.utilities.requests import RequestsWrapper

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from langchain_groq import ChatGroq

from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START

from prepare import prepare_tools, create_prompt, prepare_api_docs
from prompts import API_PLANNER_PROMPT

dotenv.load_dotenv(dotenv.find_dotenv(filename=".env"))

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Plan-and-execute"


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


# Choose the LLM that will drive the agent
# llm = ChatGroq(model_name="gemma2-9b-it", temperature=0.0)
llm = ChatGroq(model_name="llama-3.1-70b-versatile", temperature=0.0)

# tools = [TavilySearchResults(max_results=3)]
tools = prepare_tools(
    requests_wrapper,
    llm,
    allow_dangerous_requests=True,
    allowed_operations=("GET", "POST", "PUT", "DELETE", "PATCH"),
)


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


# TODO: Create prompt as state_modifier based on _create_api_controller_agent
prompt = create_prompt(spotify_api_spec, tools)


def state_modifier(state: PlanExecute):
    # print(f"state is {state}")
    # plan = state["plan"]
    # plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    plan_str = """-GET /me/playlists
- GET /playlists/{playlist_id}/tracks?limit=5"""

    return prompt.invoke({"api_docs": prepare_api_docs(plan_str, spotify_api_spec)})


tool_executer = create_react_agent(model=llm, tools=tools, state_modifier=state_modifier)


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


endpoint_descriptions = [
    f"{name} {description}" for name, description, _ in spotify_api_spec.endpoints
]

planner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        API_PLANNER_PROMPT
    ),
    ("user", """User query: {messages}
Plan:""")
])

planner = planner_prompt | ChatGroq(
    model_name="llama-3.1-70b-versatile",
    temperature=0.0
).with_structured_output(Plan)


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="""Action to perform.
If you want to respond to user, use Response.
If you need to further use tools to get the answer, use Plan."""
    )


rePlanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, \
that if executed correctly will yield the correct answer.\
Do not add any superfluous steps. \
The result of the final step should be the final answer.\
Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly.\
If no more steps are needed and you can return to the user,\
then respond with that. Otherwise, fill out the plan.\
Only add steps to the plan that still NEED to be done.\
Do not return previously done steps as part of the plan."""
)


rePlanner = rePlanner_prompt | ChatGroq(
    model_name="llama-3.1-70b-versatile",
    temperature=0.0
).with_structured_output(Act)


async def execute_step(state: PlanExecute):
    # plan = state["plan"]
    # task = plan[0]
    # plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = state["plan"][0]

    agent_response = await tool_executer.ainvoke({
        "messages": [(
            "user",
            f"""Plan: {task}
Thought:
{{agent_scratchpad}}"""
        )]
    })

    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


async def plan_step(state: PlanExecute):
    # plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    plan = await planner.ainvoke({
        "messages": state["input"],
        "endpoints": "- " + "- ".join(endpoint_descriptions)
    })

    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await rePlanner.ainvoke(state)

    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute) -> Literal["tool_executer", "__end__"]:
    if "response" in state and state["response"]:
        return "__end__"
    if "plan" in state and state["plan"] and len(state["plan"]) == 0:
        return "__end__"
    else:
        return "tool_executer"


workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("tool_executer", execute_step)

# Add a replan node
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")

# From plan we go to tool_executer
workflow.add_edge("planner", "tool_executer")

# From tool_executer, we replan
workflow.add_edge("tool_executer", "replan")

workflow.add_conditional_edges(
    "replan",
    should_end,
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

config = {"recursion_limit": 50}


async def main():
    while True:
        # command = getpass.getpass("Command: ")
        command = input("Command: ")

        inputs = {"input": command}

        async for event in app.astream(inputs, config=config):
            for k, v in event.items():
                if k != "__end__":
                    print(v)

asyncio.run(main())
