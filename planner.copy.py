import re


def create_api_controller_agent(
    api_spec: ReducedOpenAPISpec,
    requests_wrapper: RequestsWrapper,
    llm: BaseLanguageModel,
    allow_dangerous_requests: bool,
    allowed_operations: Sequence[Operation],
) -> str:
    pattern = r"\b(GET|POST|PATCH|DELETE|PUT)\s+(/\S+)*"
    matches = re.findall(pattern, plan_str)

    endpoint_names = [
        "{method} {route}".format(method=method, route=route.split("?")[0])
        for method, route in matches
    ]

    docs_str = ""
    for endpoint_name in endpoint_names:
        found_match = False

        for name, _, docs in api_spec.endpoints:
            regex_name = re.compile(re.sub("\{.*?\}", ".*", name))

            if regex_name.match(endpoint_name):
                found_match = True
                docs_str += f"== Docs for {endpoint_name} == \n{yaml.dump(docs)}\n"
                
        if not found_match:
            raise ValueError(f"{endpoint_name} endpoint does not exist.")

    agent = _create_api_controller_agent(
        base_url,
        docs_str,
        requests_wrapper,
        llm,
        allow_dangerous_requests,
        allowed_operations,
    )
    return agent