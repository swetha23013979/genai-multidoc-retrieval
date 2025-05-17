## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex
### Name:Swetha D
### Reg No:212223040222
### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.


### PROBLEM STATEMENT:
To implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and evaluate its performance.

### DESIGN STEPS:

#### STEP 1:
Load papers (PDFs) and extract vector and summary tools using get_doc_tools. Store tools in a dictionary for each paper.

#### STEP 2:
Create an object index from the tools using VectorStoreIndex. Set up a retriever to retrieve relevant information based on similarity (top 3).

#### STEP 3:
Use an agent (FunctionCallingAgentWorker) with OpenAI GPT to process the queries. Fetch results about evaluation datasets and comparisons across papers (MetaGPT, SWE-Bench, etc.).

### PROGRAM:
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=hSyW5go0v8",
    "https://openreview.net/pdf?id=VTF8yNQM66",
    "https://openreview.net/pdf?id=hSyW5go0v8"
]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "selfrag.pdf",
    "loftq.pdf",
    "swebench.pdf",
    "selfrag.pdf"
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

len(all_tools)

# define an "object" index and retriever over these tools
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

obj_retriever = obj_index.as_retriever(similarity_top_k=3)

tools = obj_retriever.retrieve(
    "Tell me about the eval dataset used in MetaGPT and SWE-Bench"
)

tools[2].metadata

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm, 
    system_prompt=""" \
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    verbose=True
)
agent = AgentRunner(agent_worker)

response = agent.query(
    "Tell me about the evaluation dataset used in LongLoRA, "
    "and then tell me about the evaluation results"
)

response = agent.query(
    "Tell me about the evaluation dataset used "
    "in MetaGPT and compare it against SWE-Bench"
)
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/42d4bbc7-b863-4d1b-bcae-914b251ba32c)

### RESULT:
Thus, The multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles is susscessfully.
