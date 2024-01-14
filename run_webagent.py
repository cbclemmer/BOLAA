import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import time

from web_run.web_env import webshopEnv
from web_run.llms import get_llm_backend, OPENAI_CHAT_MODELS, OPENAI_LLM_MODELS
import web_run.agent_arch as select_agent
from web_run.utils import get_instruction, get_env_button
from web_run.evaluate import get_file_sess_idx
from web_run.config import available_agent_names

parser = argparse.ArgumentParser(description='Parsing the input of agents, llms and llm context length.')
parser.add_argument("--agent_name", type=str, help="Name of the agent.", default="React")
parser.add_argument("--llm_name", type=str, help="Name of the llm", default="gpt-3.5-turbo")
parser.add_argument("--max_context_len", type=int, help="Maximum context length", default=1700)
args = parser.parse_args()
agent_name = args.agent_name
llm_name = args.llm_name
max_context_len = args.max_context_len

assert agent_name in available_agent_names, f"Invalid agent name. Allowed values are {available_agent_names}"

def run_one_session(idx, max_steps=50):
    env = webshopEnv()
    llm_backend = get_llm_backend(llm_name)
    llm = llm_backend.run
    saving_path = f"./execution_data/{agent.type}_{llm_name}_batch.json"
    agent = select_agent(agent_name, llm, max_context_len, saving_path)

    hulluci = 0
    # reset the environments
    action = 'reset'
    idx = f"fixed_{idx}"
    done = False
    observation, reward, done, asins, buttons = env.step(idx, action)
    env_buttons = get_env_button(buttons)
    agent.add_retrieved_item(asins)
    inst = get_instruction(observation)
    agent.new_session(idx, inst)
    
    # start planning
    if agent_name in ["PlannerReact_Webrun_Agent", "Planner_Webrun_Agent"]:
        action = 'planning'
        agent.planning()
        agent.stop = ["\n"]
        observation = agent.plan
    
    # start interaction
    for _ in range(max_steps):
        if done:
            time.sleep(1)
            agent.save()
            print("saved!")
            return reward

        action = agent.forward(observation, env_buttons).lstrip(' ') 
        print(action)
        if "No response" in action: # running too many sessions, end this session
            done = True
            return 0.0
        try:
            observation, reward, done, asins, buttons = env.step(idx, action)
            agent.add_retrieved_item(asins)
            if "Buy Now" in action:
                print(observation, reward, done, asins, buttons)
            env_buttons = get_env_button(buttons)
        except AssertionError:
            observation = 'Invalid action!'
            hulluci += 1
        if hulluci > 3:
            reward = 0.0
            done = True
        if "handle_exception" in observation: # running too many sessions, end this session
            done = True
            agent.save()
            return 0.0
    
    agent.save()
    return 0.0

def run_episodes(session_list):
    work_num = 4
    if len(session_list) > work_num and not llm_name in (OPENAI_CHAT_MODELS + OPENAI_LLM_MODELS):
        with ThreadPoolExecutor(max_workers=work_num) as executor:
            # executor.map will return a list of tuples in the same order as idxs[:10]
            results = list(executor.map(run_one_session, session_list))
            print("Done the session running!")
    else:
        for sid in session_list:
            run_one_session(sid)

file_name = f"./execution_data/webrun/{agent_name}_{llm_name}_batch.json"
if os.path.exists(file_name):
    executed_sess = get_file_sess_idx(file_name)
    session_list = [j for j in range(900) if j not in executed_sess]
    run_episodes(session_list)
else:
    run_episodes(list(range(0, 900)))


