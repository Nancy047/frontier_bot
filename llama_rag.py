from llama_index.llms.vertex import Vertex
from google.oauth2 import service_account
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.readers.gcs import GCSReader
import uvicorn
from fastapi import FastAPI, WebSocket, File, UploadFile, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

filename = "lumen-b-ctl-047-e2aeb24b0ea0.json"
credentials: service_account.Credentials = (
    service_account.Credentials.from_service_account_file(filename)
)
llm = Vertex(model="gemini-1.5-pro", project=credentials.project_id, credentials=credentials)

documents = SimpleDirectoryReader("frontier_store").load_data()
# documents = GCSReader(bucket = "frontier-data-csv", prefix="docs/", service_account_key_path= "lumen-b-ctl-047-e2aeb24b0ea0.json").load_data()     #.load_gcs_files_as_docs()
embed_model = VertexTextEmbedding(model_name= "textembedding-gecko@003")
Settings.embed_model = embed_model
vector_index = VectorStoreIndex.from_documents(documents)
chat_engine = vector_index.as_chat_engine(llm= llm, verbose = True)

@app.websocket("/ws")
async def main(websocket: WebSocket):
    await websocket.accept()
    chat_engine.memory.reset()
    while True:
        question = await websocket.receive_text()
        response = chat_engine.chat(question)
        await websocket.send_text(response.response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8097)
