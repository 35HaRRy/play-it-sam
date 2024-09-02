import re
import yaml

from typing import List, Literal, Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from langchain_community.agent_toolkits.openapi.planner_prompt import (
    PARSING_DELETE_PROMPT,
    PARSING_GET_PROMPT,
    PARSING_PATCH_PROMPT,
    PARSING_POST_PROMPT,
    PARSING_PUT_PROMPT,
)
from langchain_community.agent_toolkits.openapi.planner import (
    RequestsGetToolWithParsing,
    RequestsPostToolWithParsing,
    RequestsPatchToolWithParsing,
    RequestsPutToolWithParsing,
    RequestsDeleteToolWithParsing
)
from langchain_community.agent_toolkits.openapi.spec import ReducedOpenAPISpec
from langchain_community.utilities.requests import RequestsWrapper

from langgraph.prebuilt.chat_agent_executor import StateModifier

from prompts import API_CONTROLLER_PROMPT

Operation = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]


def prepare_tools(
    requests_wrapper: RequestsWrapper,
    llm: BaseLanguageModel,
    allow_dangerous_requests: bool,
    allowed_operations: Sequence[Operation],
) -> List[BaseTool]:
    tools: List[BaseTool] = []

    if "GET" in allowed_operations:
        get_llm_chain = PARSING_GET_PROMPT | llm
        tools.append(
            RequestsGetToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=get_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    if "POST" in allowed_operations:
        post_llm_chain = PARSING_POST_PROMPT | llm
        tools.append(
            RequestsPostToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=post_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    if "PUT" in allowed_operations:
        put_llm_chain = PARSING_PUT_PROMPT | llm
        tools.append(
            RequestsPutToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=put_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    if "DELETE" in allowed_operations:
        delete_llm_chain = PARSING_DELETE_PROMPT | llm
        tools.append(
            RequestsDeleteToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=delete_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    if "PATCH" in allowed_operations:
        patch_llm_chain = PARSING_PATCH_PROMPT | llm
        tools.append(
            RequestsPatchToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=patch_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    if not tools:
        raise ValueError("Tools not found")

    return tools


def create_prompt(
    api_spec: ReducedOpenAPISpec,
    tools: List[BaseTool],
) -> StateModifier:
    api_url = api_spec.servers[0]["url"]

    prompt = PromptTemplate(
        template=API_CONTROLLER_PROMPT,
        # input_variables=["plan_str", "agent_scratchpad"],
        input_variables=["api_docs"],
        partial_variables={
            "api_url": api_url,
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )

    return prompt


def prepare_api_docs(
    plan_str: str,
    api_spec: ReducedOpenAPISpec,
) -> StateModifier:
    pattern = r"\b(GET|POST|PATCH|DELETE|PUT)\s+(/\S+)*"
    matches = re.findall(pattern, plan_str)

    endpoint_names = [
        "{method} {route}".format(method=method, route=route.split("?")[0])
        for method, route in matches
    ]

    api_docs = ""
    for endpoint_name in endpoint_names:
        found_match = False

        for name, _, docs in api_spec.endpoints:
            regex_name = re.compile(re.sub(r"\{.*?\}", ".*", name))

            if regex_name.match(endpoint_name):
                found_match = True
                api_docs += f"== Docs for {endpoint_name} == \n{yaml.dump(docs)}\n"

        if not found_match:
            raise ValueError(f"{endpoint_name} endpoint does not exist.")

    return api_docs
