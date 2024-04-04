from typing import Dict, Union

from IPython import get_ipython
from IPython.display import display, Image
import csv
import autogen
from typing import Dict, Union
from IPython import get_ipython
from IPython.display import display, Image
import csv
import os
import autogen
import re
from utils.excute_shell_command import execute_shell_command
from utils.execute_python_file import execute_python_file
from utils.list_files_in_working_directory import list_files_in_working_directory
from utils.read_python_file import read_python_file
from utils.write_code_to_file import write_code_to_file
from prompts.prompt_text import shell_command_executor_prompt, execute_python_file_prompt, python_file_code_reader_prompt , python_coder_prompt, feature_store_prompt, admin_prompt 
 
# list_files_in_working_directory('/home/khudi/Desktop/autogen_ds/datasets/')

def check_message(message):
    if 'tool_calls' in message:
        return False
    elif not message['content'] or message['content'].lower() == 'null':
        return True
    elif re.search(r'\bTERMINAT\b', message['content'], re.IGNORECASE):
        return True
    else:
        return False



llm_config = {
    "config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}],
}


##########################################################################################
##              shell command executor
#########################################################################################

 #Ceate an AssistantAgent named "assistant"
execute_shell_command_agent = autogen.AssistantAgent(
    name="Shell Command Executor",
    # system_message="You are an expert data engineer. Your goals is to fetch data from directory (/home/khudi/Desktop/autogen_ds/datasets/final_pants_dataset_complete.csv). Do EDA on it and answer what kind of features can be made",
    system_message=shell_command_executor_prompt,
    llm_config={
        "config_list": llm_config['config_list'],  # a list of OpenAI API configurations
    },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
)

execute_shell_command_user = autogen.UserProxyAgent(
    name="Shell Command Executor User",
    max_consecutive_auto_reply=1,  # terminate without auto-reply
    human_input_mode="NEVER",
    is_termination_msg=check_message,
    code_execution_config={
        "use_docker": False,
        "work_dir": "coding",
         },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)



execute_shell_command_agent.register_for_llm(name="shell_command_executor", description="Use this tool to execute shell commands")(execute_shell_command)
execute_shell_command_user.register_for_execution(name="shell_command_executor")(execute_shell_command)


def shell_command_executor(shell_command: str) -> str:
    chat_history = execute_shell_command_user.initiate_chat(execute_shell_command_agent, message=shell_command)
    # return the last message received from the planner
    return chat_history.chat_history[-1]['content']



##########################################################################################
##             Python file executor
#########################################################################################

 #Ceate an AssistantAgent named "assistant"
python_file_executor_agent = autogen.AssistantAgent(
    name="Python File Executor",
    system_message=execute_python_file_prompt,
    llm_config={
        "config_list": llm_config['config_list'],  # a list of OpenAI API configurations
    }, 
)

python_file_executor_user = autogen.UserProxyAgent(
    name="Python File Executor User",
    max_consecutive_auto_reply=1,  # terminate without auto-reply
    human_input_mode="NEVER",
    is_termination_msg=check_message,
    code_execution_config={
        "use_docker": False,
        "work_dir": "coding",
         }, 
)



python_file_executor_agent.register_for_llm(name="execute_python_file", description="Use this tool to execute python file")(execute_python_file)
python_file_executor_user.register_for_execution(name="execute_python_file")(execute_python_file)


def python_file_exector(execute_message: str) -> str:
    chat_history = python_file_executor_user.initiate_chat(python_file_executor_agent, message=execute_message)
    # return the last message received from the planner
    return chat_history.chat_history[-1]['content']

# python_file_exector('execute file: 4_april_xgboost_forecast.py with data: /home/khudi/Desktop/autogen_ds/datasets/final_pants_dataset_complete.csv')




##########################################################################################
##             File Lister
#########################################################################################

 #Ceate an AssistantAgent named "assistant"
file_lister_agent = autogen.AssistantAgent(
    name="File Lister Agent",
    system_message=execute_python_file_prompt,
    llm_config={
        "config_list": llm_config['config_list'],  # a list of OpenAI API configurations
    }, 
)

file_lister_user = autogen.UserProxyAgent(
    name="File Lister User",
    max_consecutive_auto_reply=1,  # terminate without auto-reply
    human_input_mode="NEVER",
    is_termination_msg=check_message,
    code_execution_config={
        "use_docker": False,
        "work_dir": "coding",
         }, 
)


# file_lister_tool_die
file_lister_agent.register_for_llm(name="list_files_in_working_directory", description="Use this tool to list file in working directory")(list_files_in_working_directory)
file_lister_user.register_for_execution(name="list_files_in_working_directory")(list_files_in_working_directory)


def file_lister_exector(execute_message: str) -> str:
    chat_history = file_lister_user.initiate_chat(file_lister_agent, message=execute_message)
    # return the last message received from the planner
    return chat_history.chat_history[-1]['content']

# file_lister_exector('print the files in /home/khudi/Desktop/autogen_ds/datasets/')



##########################################################################################
##             Python File Code Reader
#########################################################################################

 #Ceate an AssistantAgent named "assistant"
python_file_code_reader_agent = autogen.AssistantAgent(
    name="Python File Code Reader Agent",
    system_message=python_file_code_reader_prompt,
    llm_config={
        "config_list": llm_config['config_list'],  # a list of OpenAI API configurations
    }, 
)

python_file_code_reader_user = autogen.UserProxyAgent(
    name="Python File Code Reader User",
    max_consecutive_auto_reply=1,  # terminate without auto-reply
    human_input_mode="NEVER",
    is_termination_msg=check_message,
    code_execution_config={
        "use_docker": False,
        "work_dir": "coding",
         }, 
)


python_file_code_reader_agent.register_for_llm(name="read_python_file", description="Use this tool to read the code in the python file")(read_python_file)
python_file_code_reader_user.register_for_execution(name="read_python_file")(read_python_file)


def python_file_code_reader(file_name: str) -> str:
    chat_history = python_file_code_reader_user.initiate_chat(python_file_code_reader_agent, message=file_name)
    # return the last message received from the planner
    return chat_history.chat_history[-1]['content']



# print(python_file_code_reader('Read the content of demo.py'))




##########################################################################################
##             Python Coder 
#########################################################################################

 #Ceate an AssistantAgent named "assistant"
python_coder_agent = autogen.AssistantAgent(
    name="Python Coder Agent",
    system_message=python_coder_prompt,
    llm_config={
        "config_list": llm_config['config_list'],  # a list of OpenAI API configurations
    }, 
)

python_coder_user = autogen.UserProxyAgent(
    name="Python Coder User",
    max_consecutive_auto_reply=1,  # terminate without auto-reply
    human_input_mode="NEVER",
    is_termination_msg=check_message,
    code_execution_config={
        "use_docker": False,
        "work_dir": "coding",
         }, 
)


# file_lister_tool_die
python_coder_agent.register_for_llm(name="write_code_to_file", description="Use this tool to write python code to a file")(write_code_to_file)
python_coder_user.register_for_execution(name="write_code_to_file")(write_code_to_file)


def python_coder(input_query: str) -> str:
    chat_history = python_coder_user.initiate_chat(python_coder_agent, message=input_query)
    # return the last message received from the planner
    return chat_history.chat_history[-1]['content']






##########################################################################################
##              Data Preparation Agent 
#########################################################################################

 #Ceate an AssistantAgent named "assistant"
feature_store_agent = autogen.AssistantAgent(
    name="Feature Store Agent",
    system_message=feature_store_prompt,
    llm_config={
        "config_list": llm_config['config_list'],  # a list of OpenAI API configurations
    }, 
)

feature_store_user = autogen.UserProxyAgent(
    name="Feature Store User",
    max_consecutive_auto_reply=1,  # terminate without auto-reply
    human_input_mode="NEVER",
    is_termination_msg=check_message,
    code_execution_config={
        "use_docker": False,
        "work_dir": "coding",
         }, 
)


# # file_lister_tool_die
# python_coder_agent.register_for_llm(name="write_code_to_file", description="Use this tool to write python code to a file")(write_code_to_file)
# python_coder_user.register_for_execution(name="write_code_to_file")(write_code_to_file)

# print(feature_store_prompt)
def feature_store(input_query: str) -> str:
    chat_history = feature_store_user.initiate_chat(feature_store_agent, message=input_query)
    # return the last message received from the planner
    return chat_history.chat_history[-1]['content']

# print(feature_store("I need to forecast the sales of march 2024. For this I need data of pant with fabric denim and region north"))


assistant = autogen.AssistantAgent(
    name="Admin",
    # system_message="You are an expert data scientist that do forecasting using xgboost algorithm. You will be given data by user. You task is to make model and output prediction and performance of model",
    system_message=admin_prompt,
    llm_config={
        "cache_seed": 41,  # seed for caching and reproducibility
        "config_list": llm_config['config_list'],  # a list of OpenAI API configurations
        "temperature": 0,  # temperature for sampling
    },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
)
# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=15,
    is_termination_msg=check_message,
    code_execution_config={
        "work_dir": "coding/",
        "use_docker": False,  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    },
)

# Register the tool signature with the assistant agent.
assistant.register_for_llm(name="shell_command_executor", description="Use this tool to execute shell commands")(shell_command_executor)
user_proxy.register_for_execution(name="shell_command_executor")(shell_command_executor)

assistant.register_for_llm(name="python_file_exector", description="Use this tool to execute python file")(python_file_exector)
user_proxy.register_for_execution(name="python_file_exector")(python_file_exector)


assistant.register_for_llm(name="file_lister_exector", description="Use this tool to list out files in the directory (same as ls in linux)")(file_lister_exector)
user_proxy.register_for_execution(name="file_lister_exector")(file_lister_exector)



assistant.register_for_llm(name="python_file_code_reader", description="Use this tool to read the code in the file (same as cat in linux)")(python_file_code_reader)
user_proxy.register_for_execution(name="python_file_code_reader")(python_file_code_reader)



 
assistant.register_for_llm(name="python_coder", description="Use this tool to write and save python code")(python_coder)
user_proxy.register_for_execution(name="python_coder")(python_coder)


assistant.register_for_llm(name="feature_store_agent", description="Use this tool to fetch the relevant data")(feature_store)
user_proxy.register_for_execution(name="feature_store_agent")(feature_store)

# # Register the tool signature with the assistant agent.
# planner.register_for_llm(name="search_engine", description="Ecommerce Search Engine")(search_product)

#     # Register the tool function with the user proxy agent.
# executor.register_for_execution(name="search_engine")(search_product)


# # the assistant receives a message from the user_proxy, which contains the task description
chat_res = user_proxy.initiate_chat(
    assistant,
    message="""Current Date: 12-12-23 \nUser: I want to forecast the sales of pant for the month of july 2025 using xgboost""",
    summary_method="reflection_with_llm",
)


