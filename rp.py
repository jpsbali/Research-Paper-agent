import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pydantic import BaseModel, Field
from typing import List, Annotated
from typing_extensions import TypedDict
import operator

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END

# === Initialize LLM ===
os.environ["GROQ_API_KEY"]= "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"
llm = ChatGroq(model="qwen-qwq-32b")

# === Arxiv Tool Setup ===
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=1000)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# === Text Splitter for Chunking Large Texts ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# === Research Paper Sections ===
RESEARCH_SECTIONS = [
    {"name": "Abstract", "description": "Write a concise academic abstract for the research paper."},
    {"name": "Introduction", "description": "Introduce the problem, explain importance, and state objectives."},
    {"name": "Literature Review", "description": "Summarize and analyze related papers from arxiv on the topic."},
    {"name": "Methodology", "description": "Describe the research methods and approach clearly."},
    {"name": "Results", "description": "Present the research findings and observations."},
    {"name": "Conclusion", "description": "Summarize key findings and contributions."},
    {"name": "Future Scope", "description": "Suggest directions for future work."},
    {"name": "References", "description": "List all cited papers in proper format."},
]

# === Pydantic Schemas ===
class Section(BaseModel):
    name: str = Field(description="Name for this section")
    description: str = Field(description="Brief overview for the section")

class Sections(BaseModel):
    sections: List[Section] = Field(description="List of sections")

# Augment LLM with structured output schema (for planning, but here we use hardcoded sections)
planner = llm.with_structured_output(Sections)

# === Graph States ===
class State(TypedDict):
    topic: str
    approach: str
    results: str
    sections: List[Section]
    completed_sections: Annotated[list, operator.add]
    arxiv_results: list  # Store raw arxiv metadata for references
    final_report: str

class WorkerState(TypedDict):
    section: Section
    topic: str
    approach: str
    results: str
    arxiv_results: list
    completed_sections: Annotated[list, operator.add]

# === Orchestrator Node ===
def orchestrator(state: State):
    # Use fixed research sections (hardcoded)
    sections = [Section(**sec) for sec in RESEARCH_SECTIONS]
    return {"sections": sections, "completed_sections": [], "arxiv_results": []}

# === Arxiv Query and Summarization ===
def arxiv_lit_review(state: WorkerState):
    topic = state["topic"]

    raw_results = arxiv_tool.invoke(topic)
    if not raw_results:
        print("Warning: No arxiv results found for topic:", topic)
        raw_results = []

    arxiv_papers = []
    references_text = []

    for doc in raw_results:
        meta = getattr(doc, "metadata", {})
        title = meta.get("title", "Unknown Title")
        authors = ", ".join(meta.get("authors", [])) if "authors" in meta else "Unknown Authors"
        summary_text = getattr(doc, "page_content", "")
        
        arxiv_papers.append({
            "title": title,
            "authors": authors,
            "summary": summary_text,
            "raw_doc": doc
        })
        
        ref = f"{authors}. \"{title}\". arXiv preprint."
        references_text.append(ref)

    state["arxiv_results"] = arxiv_papers

    chunks = []
    for paper in arxiv_papers:
        splitted = text_splitter.split_text(paper["summary"])
        chunks.extend(splitted)

    lit_summary_parts = []
    for chunk in chunks:
        summary_resp = llm.invoke([
            SystemMessage(content="You are a helpful summarizer of academic texts."),
            HumanMessage(content=f"Summarize this text concisely:\n\n{chunk}")
        ])
        if summary_resp and hasattr(summary_resp, "content"):
            lit_summary_parts.append(summary_resp.content)
        else:
            print("Warning: LLM summary response missing content")

    lit_summary_text = "\n".join(lit_summary_parts)

    lit_review_prompt = [
        SystemMessage(content="You are an academic writer drafting a literature review section."),
        HumanMessage(content=f"Based on the following summarized academic papers:\n\n{lit_summary_text}\n\nWrite a coherent Literature Review section. Mention authors and key findings, and identify gaps.")
    ]

    lit_review_section = llm.invoke(lit_review_prompt)
    if not lit_review_section or not hasattr(lit_review_section, "content"):
        raise RuntimeError("LLM literature review generation failed or missing content")

    return {
        "completed_sections": [f"## Literature Review\n\n{lit_review_section.content}"],
        "arxiv_results": arxiv_papers
    }

# === Generic LLM Call for other sections ===
def llm_call(state: WorkerState):
    section_name = state["section"].name
    section_desc = state["section"].description
    
    # If Literature Review, delegate to special node
    if section_name == "Literature Review":
        return arxiv_lit_review(state)
    
    # Build prompt based on section and state context
    prompt_map = {
        "Abstract": f"You are a research writer. Given the topic, approach, and results, write a concise abstract.\nTopic: {state['topic']}\nApproach: {state['approach']}\nResults: {state['results']}\nAbstract:",
        "Introduction": f"Write an Introduction for the topic: {state['topic']} explaining the importance and objectives based on the approach: {state['approach']}.",
        "Methodology": f"Describe the methodology used for the approach: {state['approach']}.",
        "Results": f"Explain the results of the research: {state['results']}.",
        "Conclusion": f"Summarize the key findings and contributions on the topic: {state['topic']}.",
        "Future Scope": f"Suggest future research directions for the topic: {state['topic']} based on the approach: {state['approach']}.",
        "References": "",  # Will be generated by separate node
    }
    
    # If References, generate from stored arxiv_results
    if section_name == "References":
        arxiv_papers = state.get("arxiv_results", [])
        refs = []
        for paper in arxiv_papers:
            refs.append(f"- {paper['authors']}. \"{paper['title']}\". arXiv preprint.")
        ref_text = "\n".join(refs)
        return {"completed_sections": [f"## References\n\n{ref_text}"]}
    
    prompt = prompt_map.get(section_name, f"Write the {section_name} section for the topic {state['topic']}.")
    
    llm_response = llm.invoke([
        SystemMessage(content=f"Write a report section titled '{section_name}' in markdown format. {section_desc}"),
        HumanMessage(content=prompt)
    ])
    
    return {"completed_sections": [f"## {section_name}\n\n{llm_response.content}"]}

# === Synthesizer Node ===
def synthesizer(state: State):
    # Join all completed sections with Markdown separators
    final_report = "\n\n---\n\n".join(state["completed_sections"])
    return {"final_report": final_report}

# === Assign Workers Node ===
def assign_workers(state: State):
    # Create worker calls for each section
    return [Send("llm_call", {"section": s, "topic": state["topic"], "approach": state["approach"], "results": state["results"], "arxiv_results": state.get("arxiv_results", [])}) for s in state["sections"]]

# === Build Workflow ===
orchestrator_worker_builder = StateGraph(State)
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

orchestrator_worker = orchestrator_worker_builder.compile()

# === Streamlit UI ===

def generate_research_paper(topic, approach, results):
    initial_state = {
        "topic": topic,
        "approach": approach,
        "results": results,
    }
    final_state = orchestrator_worker.invoke(initial_state)
    return final_state["final_report"]

def main():
    st.title("Research Paper Generator with LangGraph + Arxiv")
    st.write(
        """
        Enter your research details below. 
        This app will generate a structured research paper with literature review and references using AI.
        """
    )
    
    topic = st.text_input("Research Topic", "Agentic AI and Retrieval-Augmented Generation Systems")
    approach = st.text_area("Research Approach", 
                           "We developed an agentic AI pipeline using LangChain and LangGraph with arxiv-based literature review.")
    results = st.text_area("Research Results", 
                          "The system successfully generates research papers with well-structured sections and relevant references.")
    
    if st.button("Generate Research Paper"):
    with st.spinner("Generating paper... This may take a minute."):
        try:
            # Call your paper generation function
            paper_md_raw = generate_research_paper(topic, approach, results)
            
            # Clean the output if it contains <think> tags
            if "</think>" in paper_md_raw:
                paper_md = paper_md_raw.split("</think>")[-1].strip()
            else:
                paper_md = paper_md_raw.strip()
            
            st.success("Research paper generated successfully!")
            
            # Display cleaned markdown output
            st.markdown(paper_md)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
