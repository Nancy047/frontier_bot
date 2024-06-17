import os
import json
import base64
import pandas as pd
from fastapi import FastAPI, WebSocket
import uvicorn
from pandasai import SmartDataframe
from langchain_google_vertexai import VertexAI
from pandasai import Agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from google.cloud import storage

app = FastAPI()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "lumen-b-ctl-047-e2aeb24b0ea0.json"

llm = VertexAI(
    model_name="gemini-1.5-pro", temperature=0.5, max_output_tokens=8192
)

template = """You are provided with a data containing customer details or
    case details of customer. Using the provided json, give a concise summary of the data. Try to be concise
    Highlight the specifics and the numbers.
    
    Give me a detailed inference based on this data. 

    {chat_history}
    df input: {human_input}
    Response:"""
    
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client.from_service_account_json('lumen-b-ctl-047-e2aeb24b0ea0.json')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

csv_list = ["may_clearingcodes_dispositions_field_updated 1.csv", "may_dispatches_celonis_updated 1.csv", "may_dispositions_tso_updated 1.csv", "may_workflow_tso.csv", "result.csv"]
for csv in csv_list:
    download_blob('frontier-data-csv', csv, csv)

clearingcodes_df = pd.read_csv("may_clearingcodes_dispositions_field_updated 1.csv")
dispatches_celonis_df = pd.read_csv("may_dispatches_celonis_updated 1.csv")
disposition_tso_df = pd.read_csv("may_dispositions_tso_updated 1.csv")
workflow_tso_df = pd.read_csv("may_workflow_tso.csv")
hellox_df = pd.read_csv("result.csv")

pai_agent = Agent(
    [clearingcodes_df, dispatches_celonis_df, disposition_tso_df, workflow_tso_df, hellox_df],
    memory_size=10,
    description="""You are a data analysis agent. Your main goal is to help non-technical users to analyze data.""",
    config={"llm": llm, "enable_cache": False},
)
df_agent = create_pandas_dataframe_agent(llm, hellox_df, verbose=True)
pai_agent.clear_memory()

async def data_summary(df_json: str, websocket: WebSocket):
    memory.clear()
    print("memory is cleared")
    response = llm_chain.run(df_json)
    print("===================")
    print(f"Response: {response}")
    return response

def df_to_json(df):
    # Convert DataFrame column names to a more JSON-friendly format (e.g., lowercase with underscores)
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    json_output = []
    unique_id_col = df.columns[0]  # Dynamically determine the unique identifier column

    for _, row in df.iterrows():
        entry = {unique_id_col: str(row[unique_id_col])}  # Convert to string

        # Loop through all other columns to add them as attributes
        for col in df.columns[1:]:
            if isinstance(row[col], pd.Timestamp):
                print("converting Datetime to ISO")
                entry[col] = row[
                    col
                ].isoformat()  # Convert Timestamp to ISO format string
            else:
                entry[col] = row[col]

        json_output.append(entry)

    return json_output


def png_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")


async def pandas_response(query: str):
    try:
        try:
            response = pai_agent.chat(query)
            print(f"Pandasai response: \n{response}")
            if str(response).startswith("Unfortunately"):
                print("response starts with Unfortunately")
                raise ValueError("PandasAI agent couldn't give response")
            result = pai_agent.last_code_executed
            json_result = result.split("\n")[::-1][0].split(" = ")[1].replace("'", '"')
            print(f"Last Code Executed: \n{json_result}")
            try:
                json_result = json.loads(json_result)
            except:
                json_result = json.loads(json_result.split(",")[0] + "}")
            if json_result["type"] == "string":
                print("String Response")
                return {"image": None, "description": response, "table": None}
            elif json_result["type"] == "number":
                print("Numerical Response")
                return {"image": None, "description": f"{response}", "table": None}
            elif json_result["type"] == "dataframe":
                print("Dataframe Response")
                df_json = df_to_json(response)
                return {"image": None, "description": None, "table": df_json}
            elif json_result["type"] == "plot":
                print("Plot Response")
                base64_img = png_to_base64(response)
                return {"image": base64_img, "description": None, "table": None}
            else:
                print(f"Unhandled Response Type - {json_result['type']}")
                try:
                    return {"image": None, "description": response, "table": None}
                except:
                    return {
                        "image": None,
                        "description": "Unexpected Response Type",
                        "table": None,
                    }
        except:
            try:
                print("Entering Dataframe Agent")
                response = df_agent.invoke(query)
                print(f"df_agent response: \n{response}")
                return {"image": None, "description": response["output"], "table": None}
            except Exception as e:
                return {
                    "image": None,
                    "description": f"Error generating response from the model - {e}",
                    "table": None,
                }
    except Exception as e:
        print("No response generated.")
        return {
            "image": None,
            "description": f"Error generating response from the model - {e}",
            "table": None,
        }


@app.websocket("/pd")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    memory.clear()
    while True:
        data = await websocket.receive_text()
        response = await pandas_response(data)
        print(type(response))
        if response["table"] != None:
            # await websocket.send_text("Do you want to summarize?")
            # data = await websocket.receive_text()
            # if data.lower() == "yes":
            print("Summarizing")
            # summary = await data_summary(response["table"], websocket)
            summary = await data_summary(json.dumps(response["table"]), websocket)
            await websocket.send_text(summary)
        try:
            await websocket.send_json(response)
        except:
            await websocket.send_json(
                {"image": None, "description": "Type Handling Error", "table": None}
            )


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
