from dotenv import load_dotenv
load_dotenv()
import re
import os
import shutil
import base64
import json
import hashlib
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
## document loader
from docling.document_converter import DocumentConverter
## document chunking
from langchain_experimental.text_splitter import SemanticChunker
## embedding model
from langchain_huggingface import HuggingFaceEmbeddings


## vector database
from langchain_community.vectorstores import Chroma
## keyword search retriever algorithm
from langchain_community.retrievers import BM25Retriever
## combinig vector and keyword--hybrid
from langchain_classic.retrievers import EnsembleRetriever
## LLM model
from langchain_google_genai import ChatGoogleGenerativeAI
## prompt template
from langchain_core.prompts import ChatPromptTemplate

## Reranking compression

from langchain_classic.retrievers import(ContextualCompressionRetriever)
from langchain_classic.retrievers.document_compressors import ( CrossEncoderReranker)
from langchain_community.cross_encoders import (HuggingFaceCrossEncoder)
from typing import Optional, Literal

## pydantic output form
class MoleculeNode(BaseModel):

    label: str = Field(
        description="The exact or near-exact text used for this molecule's node "
        "in mermaid_diagram (e.g. the formula or name shown in that node)"
    )

    smiles: str = Field(
        description="Valid SMILES string for this molecule's bonded structure"
    )


class EducationalResponse(BaseModel):

    explanation: str = Field(
        description="Detailed explanation"
    )

    subject: Literal[
        "chemistry",
        "physics",
        "mathematics"
    ]

    content_types:list[ Literal[
        "text",
        "equation",
        "chemical_equation",
        "structure",
        "flowchart",
        "table"
    ]]

    equation_latex: Optional[list[str]] = Field(
        default=None,
        description="Mathematical or physics equation in LaTeX"
    )

    reaction_mhchem: Optional[list[str]] = Field(
        default=None,
        description="Chemical equation in mhchem format"
    )



    mermaid_diagram: Optional[str] = Field(
        default=None,
        description="Flowchart or process diagram in Mermaid format"
    )

    node_structures: Optional[list[MoleculeNode]] = Field(
        default=None,
        description="One entry per distinct molecule that appears as a node in "
        "mermaid_diagram, each with that node's label and its SMILES, so every "
        "molecule in the flowchart can be drawn as a real bonded structure "
        "instead of plain formula text"
    )

    catalyst: Optional[str] = None

    condition: Optional[str] = None

    table_markdown: Optional[list[str]] = None
## creating an object for pydantic model

parser = PydanticOutputParser(
    pydantic_object=EducationalResponse
)

format_instructions = parser.get_format_instructions()

## document loader (docling preserves headings, tables, and reading order
## far better than plain-text docx extraction, which helps recover formulas
## and reaction diagrams instead of flattening them into jumbled text)

docling_converter = DocumentConverter()
docling_result = docling_converter.convert('updatedimage_EQUATIONS.docx')

document = [
    Document(
        page_content=docling_result.document.export_to_markdown(),
        metadata={"source": "EQUATIONS.docx"}
    )
]

## LLM (moved up so it's available for image captioning below)

llm=ChatGoogleGenerativeAI(
    model='gemini-flash-lite-latest',
    temperature=0.2,
    api_key=os.getenv('GOOGLE_API_KEY')
)

## Extract embedded chemical-bonding images and caption them for retrieval
##
## Docling pulls each embedded picture out of the docx as a PIL image. We save
## it to disk and ask the (vision-capable) Gemini model for a short factual
## caption describing what it shows; that caption is what actually gets
## embedded/indexed, so a query about the reaction/structure in the image can
## retrieve it like any other chunk. The saved file path travels in the
## Document's metadata so the real image can be handed back to the model
## (multimodally) at answer time.

IMAGE_DIR = "extracted_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

## Captioning an image costs a real LLM round trip. Without caching, every
## single script run re-captioned every image from scratch (on top of the
## main answer call), which is what was making every run slow. Cache by
## content hash so an image only ever gets captioned once.
CAPTION_CACHE_PATH = os.path.join(IMAGE_DIR, "captions_cache.json")

def load_caption_cache():
    if os.path.exists(CAPTION_CACHE_PATH):
        try:
            with open(CAPTION_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_caption_cache(cache):
    with open(CAPTION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def extract_text(content):
    # Gemini sometimes returns .content as a plain string and sometimes as a
    # list of content blocks (e.g. [{"type": "text", "text": "..."}]).
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return "".join(parts).strip()
    return str(content).strip()

def caption_image(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    message = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                "This image is a chemical reaction/bonding diagram from a "
                "chemistry textbook. In 2-3 sentences, factually describe "
                "exactly what it shows: the starting compound(s), each "
                "product, and the reagents/conditions for each step. Use "
                "the actual formulas/names visible in the image. This "
                "description will be used to search for this image later."
            ),
        },
        {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
    ])
    return extract_text(llm.invoke([message]).content)

caption_cache = load_caption_cache()
image_documents = []
for i, picture in enumerate(docling_result.document.pictures):
    if picture.image is None:
        continue
    image_path = os.path.join(IMAGE_DIR, f"image_{i}.png")
    picture.image.pil_image.convert("RGB").save(image_path)
    img_hash = file_hash(image_path)
    if img_hash in caption_cache:
        caption = caption_cache[img_hash]
    else:
        caption = caption_image(image_path)
        caption_cache[img_hash] = caption
    image_documents.append(
        Document(
            page_content=caption,
            metadata={
                "source": "EQUATIONS.docx",
                "type": "image",
                "image_path": image_path,
            },
        )
    )
save_caption_cache(caption_cache)

## Data cleaning

def clean_text(text):
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text

##structerd Documents

processed_docs = []

for doc in document:

    cleaned_text = clean_text(
        doc.page_content
    )

    structured_doc = Document(
        page_content=cleaned_text,
        metadata={

            "source":'EQUATIONS.DOCX',
            "page": doc.metadata.get("page")
        }
    )

    processed_docs.append(
        structured_doc
    )


## Embeddings_model

embedding_model=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

## chunking (semantic chunking: splits on meaning shifts between sentences,
## using the embedding model above, instead of a fixed character count)

## Default percentile (95) produced only ~8 very large, topic-mixed chunks
## for this document (up to 10k+ chars each) — a named reaction like
## "Sandmeyer" ended up sharing a chunk with several unrelated topics, which
## diluted it enough that the cross-encoder scored the whole chunk as
## irrelevant to a direct "explain Sandmeyer reaction" query even though the
## answer was right there in it. Lowering the threshold produces more, smaller,
## topically-focused chunks so each named reaction gets scored on its own
## merits instead of being buried inside a much larger, unrelated block.
splitter=SemanticChunker(
    embedding_model,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=80,
)

chunks=splitter.split_documents(processed_docs)
chunks = [c for c in chunks if len(c.page_content.strip()) > 20]

## Add the image captions as their own retrievable chunks (already short,
## no further splitting needed) so a query can match and retrieve an image.
chunks = chunks + image_documents

## Store in vector databse

## Chroma.from_documents() appends to any existing persisted collection instead
## of replacing it, so re-running this script kept re-adding the same chunks
## and silently duplicating the index. Clear it first so every run starts from
## a clean, single copy of the current document's chunks.
shutil.rmtree("./chroma_db", ignore_errors=True)

vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

## vector retriever

## Cutting k/top_n down (5/4) to fight "images for every query" backfired:
## with SemanticChunker producing only ~8 large, topically-mixed chunks, a
## small k sometimes excluded the actually-relevant chunk (e.g. Sandmeyer)
## from the candidate pool entirely, before it ever got a chance to be
## scored. Precision is now handled downstream by RELEVANCE_THRESHOLD /
## IMAGE_RELEVANCE_THRESHOLD (which score each candidate against the actual
## question), so retrieval itself should cast a wide net across this small
## corpus rather than pre-filtering by a blunt cutoff.
vector_retriever=vectorstore.as_retriever(search_kwargs={'k':10})

## keyword retriever+

keyword_retriever=BM25Retriever.from_documents(chunks)

keyword_retriever.k=10

## hybrid retriever(vector_store+keyword_retriever)

ensemble_retriever=EnsembleRetriever(retrievers=[vector_retriever,keyword_retriever],weights=[0.5,0.5])

## Reranker

cross_encoder = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-base"
)

reranker = CrossEncoderReranker(
    model=cross_encoder,
    top_n=8
)


##contextcompression-it retrieve and compress the answer

compression_retriever = (
    ContextualCompressionRetriever(
        base_retriever=ensemble_retriever,
        base_compressor=reranker
    )
)



## prompt template

prompt = ChatPromptTemplate.from_template(
r"""
You are an expert educational tutor for Chemistry, Physics, and Mathematics.

Answer using the provided context. Ground reactions, molecules, and diagrams
in what the context actually says — interpreting or cleaning up a messy or
prose-only description into proper notation is fine, adding content the
context never mentions is not.

If the context doesn't answer the question: content_types=["text"],
explanation="I don't know based on the provided document.", every other
field null.

Otherwise fill in whichever fields below fit the answer — they're
independent of each other, not alternatives. content_types just lists which
ones are present, and every type you list there must actually be filled in:
if you include "chemical_equation", reaction_mhchem or mermaid_diagram must
be non-empty; if you include "structure", node_structures must be non-empty;
if you include "equation", equation_latex must be non-empty. Never list a
type without also populating its field.

- explanation: plain prose describing the answer.

- reaction_mhchem: list of reactions, one entry per distinct reaction, each
  written as reactants -> products in mhchem, e.g.
  \ce{{CH3COOH + C2H5OH -> CH3COOC2H5 + H2O}}
  Keep catalysts/reagents out of the equation itself — put those in
  catalyst/condition instead. If a transformation involves multiple
  dependent steps or branches by condition, use mermaid_diagram rather than
  chaining several arrows into one entry. If the question asks for a
  reaction/preparation and the context only describes it in prose (reactants,
  reagents, and product named in words, not yet as a formula), convert that
  prose into reaction_mhchem or mermaid_diagram yourself — don't fall back
  to explanation-only just because it isn't already written as an equation.

- catalyst / condition: the reagent, catalyst, or condition (temperature,
  light, pressure, etc.) driving the reaction, when the context names one.

- mermaid_diagram: a Mermaid flowchart for a multi-step or branching
  pathway — one node per compound, one edge per step, reagents as edge
  labels.

- node_structures: list of {{label, smiles}}. ONLY populate this when the
  question itself asks about a molecule's structure, shape, or bonding (e.g.
  "what is the structure of X", "draw/show the bonding of X", "what does X
  look like"). If the question is about a reaction, mechanism, preparation,
  or property WITHOUT asking for structure specifically, leave this null and
  do NOT include "structure" in content_types — a reaction/flowchart answer
  does not need every node drawn as a picture unless structure was actually
  requested. Use "*" for a generic/unspecified substituent (a bare "R" is not
  valid SMILES).

- equation_latex: list of math/physics formulas, in LaTeX.

- table_markdown: list of tables copied from the context, if relevant.

- subject: "chemistry", "physics", or "mathematics".

Some context items are real images (chemical bonding/reaction diagrams), not
text — they're attached directly below as pictures. The user will NEVER see
the picture itself, only your structured fields, so you must fully transcribe
everything in it into text: every reaction (reaction_mhchem or
mermaid_diagram — use mermaid_diagram if it shows branching/multi-step
pathways), every reagent/condition on every arrow, every molecule's structure
(node_structures) if the question asks about structure, and every labeled
product name. Do not summarize the image loosely or describe it in vague
prose ("this diagram shows a reduction reaction") — extract its complete
content the same way you would a formula box, as if you were redrawing it in
words for someone who cannot see it at all.

{format_instructions}

Context:
{context}

Question:
{input}

"""
).partial(
    format_instructions=format_instructions
)

## structured LLM (vision-capable — Gemini accepts image input directly)

structured_llm = llm.with_structured_output(
    EducationalResponse
)

## Custom answer step (replaces create_stuff_documents_chain +
## create_retrieval_chain): retrieved documents can be plain text chunks OR
## image-caption chunks (metadata type="image"). Text chunks are stuffed into
## the prompt as before; for any image chunk retrieved, the actual saved
## image file is base64-encoded and attached to the model call so it can see
## and interpret the diagram directly, instead of relying only on its
## precomputed caption.
##
## RELEVANCE_THRESHOLD: with only ~10 documents in this corpus (text chunks +
## image captions combined), k=10/top_n=7 meant the reranker was returning
## almost the entire corpus - images included - for every query, even ones
## unrelated to this document. CrossEncoderReranker only sorts and truncates;
## it doesn't expose a relevance floor. So we re-score each candidate
## ourselves and drop anything that isn't actually relevant to this specific
## question. Calibrated empirically: this cross-encoder scores a genuinely
## relevant match ~0.85-0.9 and a totally unrelated query/document pair
## ~0.0 — but a legitimately relevant match can still score much lower than
## 0.85 when the question's wording doesn't closely echo the chunk's wording
## (e.g. "what is the structure of aniline" against a chunk that discusses
## aniline only as a reaction product, not as a dedicated structure
## description). RELEVANCE_THRESHOLD is kept low to avoid rejecting those
## legitimate-but-loosely-worded matches; only images get a strict threshold,
## since attaching a picture should be a deliberate, high-confidence match.
RELEVANCE_THRESHOLD = 0.05
IMAGE_RELEVANCE_THRESHOLD = 0.3

def answer_question(question):
    docs = compression_retriever.invoke(question)

    if docs:
        scores = cross_encoder.score([(question, d.page_content) for d in docs])
        filtered = []
        for d, s in zip(docs, scores):
            threshold = IMAGE_RELEVANCE_THRESHOLD if d.metadata.get("type") == "image" else RELEVANCE_THRESHOLD
            if s >= threshold:
                filtered.append(d)
        docs = filtered

    if not docs:
        # Nothing in the document is actually relevant to this question —
        # answer that directly instead of calling the LLM with weak/unrelated
        # context (which is what was producing irrelevant answers before).
        return EducationalResponse(
            explanation="I don't know based on the provided document.",
            subject="chemistry",
            content_types=["text"],
        )

    text_docs = [d for d in docs if d.metadata.get("type") != "image"]
    image_docs = [d for d in docs if d.metadata.get("type") == "image"]

    context_text = "\n\n".join(d.page_content for d in text_docs)
    prompt_text = prompt.format(context=context_text, input=question)

    content = [{"type": "text", "text": prompt_text}]
    for d in image_docs:
        image_path = d.metadata.get("image_path")
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({"type": "image_url", "image_url": f"data:image/png;base64,{b64}"})

    # Images are used only to let the model SEE and read the diagram — the
    # raw picture is never handed back to the user. Everything it shows must
    # come through as text via the structured fields above (see the
    # "Some context items are real images..." prompt instruction).
    result = structured_llm.invoke([HumanMessage(content=content)])

    return result

## Ask questions

## Previously this was a hardcoded string literal, so the script always
## answered the same fixed question ("show Chemical Properties of
## nitrobenzene.") no matter what you actually wanted to ask — which is why
## every run looked identical. Take the real question from the user instead.
user_question = input("Ask a question: ").strip()
answer = answer_question(user_question)

## Render explanation + equation + reaction + structure into one image

def render_answer(answer, output_dir="outputs"):
    import re
    from datetime import datetime
    from io import BytesIO

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(output_dir, exist_ok=True)

    font_path = font_manager.findfont("DejaVu Sans")
    header_font = ImageFont.truetype(font_path, 24)
    body_font = ImageFont.truetype(font_path, 18)

    CANVAS_WIDTH = 900
    PADDING = 30

    # Shared by every mhchem/prose-cleanup helper below — was previously
    # redefined in three separate places with the same regex.
    SUBSCRIPT_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    def strip_ce(mhchem):
        return re.sub(r"^\\ce\{(.*)\}$", r"\1", mhchem.strip())

    def mhchem_to_text(mhchem):
        text = strip_ce(mhchem)
        text = text.replace("<=>", "⇌").replace("->", "→").replace("<-", "←")
        text = re.sub(
            r"([A-Za-z\)\]])(\d+)",
            lambda m: m.group(1) + m.group(2).translate(SUBSCRIPT_MAP),
            text,
        )
        return text

    def clean_prose(text):
        # The model is instructed to keep explanation as plain prose, but
        # sometimes leaks LaTeX-ish formula fragments into it anyway (e.g.
        # "$C_{6}H_{5}-NO_{2}$"). Strip the LaTeX delimiters/braces so it
        # reads as plain text instead of raw markup.
        text = text.replace("$", "")
        text = re.sub(r"_\{(\d+)\}", lambda m: m.group(1).translate(SUBSCRIPT_MAP), text)
        text = re.sub(r"_([0-9])", lambda m: m.group(1).translate(SUBSCRIPT_MAP), text)
        text = re.sub(r"\^\{([^{}]*)\}", lambda m: m.group(1), text)
        text = re.sub(r"\\[a-zA-Z]+", "", text)
        return text

    def mhchem_arrow_count(mhchem):
        body = strip_ce(mhchem)
        return len(re.findall(r"->|<=>", body))

    def mhchem_to_mermaid(mhchem):
        # Safety net: the model is instructed to keep reaction_mhchem to a
        # single arrow and use mermaid_diagram for anything multi-step, but
        # it doesn't always comply. If it chains multiple arrows into one
        # reaction_mhchem anyway, convert that chain into a flowchart instead
        # of rendering the broken concatenated string.
        body = strip_ce(mhchem)
        tokens = re.split(r"(->\s*(?:\[[^\]]*\])?|<=>)", body)
        segments = [t.strip() for t in tokens[0::2]]
        arrows = [t.strip() for t in tokens[1::2]]
        if len(segments) < 2:
            return None
        lines = ["flowchart TD"]
        for i in range(len(segments) - 1):
            reagent_match = re.search(r"\[([^\]]*)\]", arrows[i]) if i < len(arrows) else None
            reagent = reagent_match.group(1) if reagent_match else ""
            src_label = segments[i].replace('"', "'") or f"Step {i}"
            dst_label = segments[i + 1].replace('"', "'") or f"Step {i + 1}"
            edge = f'|"{reagent}"|' if reagent else ""
            lines.append(f'N{i}["{src_label}"] -->{edge} N{i + 1}["{dst_label}"]')
        return "\n".join(lines)

    def render_latex(latex):
        try:
            fig = plt.figure(figsize=(8, 1.4))
            fig.patch.set_alpha(0)
            fig.text(0.02, 0.5, f"${latex}$", fontsize=24, va="center", ha="left")
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2, transparent=True, dpi=150)
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf).convert("RGBA")
        except Exception:
            plt.close("all")
            return None

    def render_smiles(smiles):
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # "R" is organic chemistry's informal placeholder for a generic
            # substituent, but it isn't valid SMILES; retry treating it as a
            # wildcard atom ("*") in case the model used it anyway.
            sanitized = re.sub(r"(?<![A-Za-z])R(?![a-z])", "*", smiles)
            if sanitized != smiles:
                mol = Chem.MolFromSmiles(sanitized)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=(350, 350)).convert("RGBA")

    def render_mermaid(mermaid_str, node_smiles=None):
        try:
            import textwrap as _textwrap

            import networkx as nx
            import numpy as np
            from matplotlib.offsetbox import AnnotationBbox, OffsetImage

            edge_pattern = re.compile(
                r"([A-Za-z0-9_]+)\s*(\[[^\]]*\]|\([^\)]*\)|\{[^\}]*\})?\s*-->\s*"
                r"(\|[^|]*\|)?\s*([A-Za-z0-9_]+)\s*(\[[^\]]*\]|\([^\)]*\)|\{[^\}]*\})?"
            )

            def clean_label(text):
                text = text.strip()
                if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
                    text = text[1:-1].strip()
                return text

            def wrap_label(text, width=16):
                return "\n".join(_textwrap.wrap(text, width=width)) or text

            graph = nx.DiGraph()
            labels = {}
            edge_labels = {}

            for line in mermaid_str.splitlines():
                line = line.strip()
                if not line or line.lower().startswith(("flowchart", "graph")):
                    continue
                m = edge_pattern.match(line)
                if not m:
                    continue
                src, src_shape, edge_lbl, dst, dst_shape = m.groups()
                if src_shape:
                    labels[src] = clean_label(src_shape[1:-1])
                if dst_shape:
                    labels[dst] = clean_label(dst_shape[1:-1])
                labels.setdefault(src, src)
                labels.setdefault(dst, dst)
                graph.add_edge(src, dst)
                if edge_lbl:
                    edge_labels[(src, dst)] = clean_label(edge_lbl.strip("|"))

            if graph.number_of_nodes() == 0:
                return None

            resolved_structures = {}
            if node_smiles:
                for node_id, disp in labels.items():
                    disp_l = disp.lower()
                    for entry_label, smi in node_smiles.items():
                        entry_l = entry_label.lower().strip()
                        if not entry_l:
                            continue
                        if entry_l in disp_l or disp_l in entry_l:
                            struct_img = render_smiles(smi)
                            if struct_img:
                                resolved_structures[node_id] = struct_img
                            break

            has_structures = bool(resolved_structures)
            layer_step = 3.6 if has_structures else 2.6
            row_step = 2.8 if has_structures else 1.8

            try:
                generations = [list(gen) for gen in nx.topological_generations(graph)]
            except Exception:
                generations = [list(graph.nodes())]

            pos = {}
            for layer_idx, layer_nodes in enumerate(generations):
                n = len(layer_nodes)
                for i, node in enumerate(layer_nodes):
                    pos[node] = (layer_idx * layer_step, -(i - (n - 1) / 2) * row_step)

            max_layer_size = max(len(g) for g in generations)
            fig, ax = plt.subplots(
                figsize=(
                    max(5, len(generations) * layer_step),
                    max(3, max_layer_size * row_step),
                )
            )
            ax.axis("off")

            shrink = 34 if has_structures else 22
            for u, v in graph.edges():
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                layer_gap = abs(generations.index(next(g for g in generations if u in g))
                                 - generations.index(next(g for g in generations if v in g)))
                connectionstyle = "arc3,rad=0.25" if layer_gap > 1 else "arc3,rad=0.0"
                ax.annotate(
                    "", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="->", color="black", lw=1.5,
                        shrinkA=shrink, shrinkB=shrink, connectionstyle=connectionstyle,
                    ),
                )
                lbl = edge_labels.get((u, v))
                if lbl:
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(
                        mx, my, wrap_label(lbl, width=20), fontsize=8, ha="center", va="center",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
                    )

            for node, (x, y) in pos.items():
                struct_img = resolved_structures.get(node)
                if struct_img is not None:
                    target_px = 110
                    zoom = target_px / struct_img.width
                    imagebox = OffsetImage(np.array(struct_img), zoom=zoom)
                    ab = AnnotationBbox(
                        imagebox, (x, y + 0.35),
                        frameon=True, pad=0.35,
                        bboxprops=dict(edgecolor="#4f46e5", boxstyle="round,pad=0.3", facecolor="white"),
                    )
                    ax.add_artist(ab)
                    ax.text(
                        x, y - 0.75, wrap_label(labels.get(node, node), width=18),
                        fontsize=8, ha="center", va="top",
                    )
                else:
                    ax.text(
                        x, y, wrap_label(labels.get(node, node), width=16), fontsize=10,
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef2ff", edgecolor="#4f46e5"),
                    )

            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            x_margin = 1.6 if has_structures else 1.3
            y_margin = 1.6 if has_structures else 1.1
            ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
            ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.3, dpi=150, facecolor="white")
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf).convert("RGBA")
        except Exception:
            plt.close("all")
            return None

    def wrap_text(text, font, max_width):
        draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        words = text.split()
        lines, current = [], ""
        for word in words:
            trial = f"{current} {word}".strip()
            if draw.textlength(trial, font=font) <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def render_structure_grid(node_structures):
        cards = []
        for ns in node_structures:
            img = render_smiles(ns.smiles)
            if img:
                target = 220
                ratio = target / img.width
                img = img.resize((target, int(img.height * ratio)))
                cards.append((ns.label, img))
        if not cards:
            return None
        cols = min(3, len(cards))
        rows = (len(cards) + cols - 1) // cols
        cell_w = 240
        cell_h = 260
        grid = Image.new("RGBA", (cell_w * cols, cell_h * rows), "white")
        gdraw = ImageDraw.Draw(grid)
        for i, (label, img) in enumerate(cards):
            cx = (i % cols) * cell_w
            cy = (i // cols) * cell_h
            img_x = cx + (cell_w - img.width) // 2
            grid.paste(img, (img_x, cy + 5), img)
            for j, line in enumerate(wrap_text(label, body_font, cell_w - 10)):
                gdraw.text((cx + 10, cy + img.height + 15 + j * (body_font.size + 4)), line, font=body_font, fill="black")
        return grid

    def render_table(md_table):
        rows = []
        for line in md_table.strip().splitlines():
            line = line.strip().strip("|")
            if not line or re.fullmatch(r"[\s:\-|]+", line):
                continue
            rows.append([c.strip() for c in line.split("|")])
        if not rows:
            return None
        n_cols = max(len(r) for r in rows)
        rows = [r + [""] * (n_cols - len(r)) for r in rows]
        col_widths = []
        tmp_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        for c in range(n_cols):
            widest = max(tmp_draw.textlength(r[c], font=body_font) for r in rows)
            col_widths.append(int(widest) + 24)
        row_h = body_font.size + 20
        table_w = sum(col_widths)
        table_h = row_h * len(rows)
        img = Image.new("RGBA", (table_w, table_h), "white")
        d = ImageDraw.Draw(img)
        y = 0
        for ri, r in enumerate(rows):
            x = 0
            for ci, cell in enumerate(r):
                d.rectangle([x, y, x + col_widths[ci], y + row_h], outline="#999999")
                if ri == 0:
                    d.rectangle([x, y, x + col_widths[ci], y + row_h], fill="#eef2ff", outline="#999999")
                d.text((x + 10, y + 10), cell, font=body_font, fill="black")
                x += col_widths[ci]
            y += row_h
        return img

    blocks = []  # ("text", lines, font) or ("image", PIL Image)

    header_label = " / ".join(t.replace("_", " ").title() for t in answer.content_types)
    header = f"{answer.subject.title()} \u2014 {header_label}"
    blocks.append(("text", [header], header_font))

    # Display order: explanation -> chemical reaction(s) -> molecular
    # structure(s) -> flowchart -> table -> math/physics equation(s). Every
    # populated field is shown; none of these are exclusive of one another.

    if answer.explanation:
        blocks.append(("text", wrap_text(clean_prose(answer.explanation), body_font, CANVAS_WIDTH - 2 * PADDING), body_font))

    for reaction in (answer.reaction_mhchem or []):
        if mhchem_arrow_count(reaction) > 1:
            node_smiles = (
                {ns.label: ns.smiles for ns in answer.node_structures}
                if answer.node_structures else None
            )
            synthesized_mermaid = mhchem_to_mermaid(reaction)
            flow_img = render_mermaid(synthesized_mermaid, node_smiles=node_smiles) if synthesized_mermaid else None
            if flow_img:
                blocks.append(("image", flow_img))
            else:
                blocks.append(("text", wrap_text(mhchem_to_text(reaction), body_font, CANVAS_WIDTH - 2 * PADDING), body_font))
        else:
            blocks.append(("text", wrap_text(mhchem_to_text(reaction), body_font, CANVAS_WIDTH - 2 * PADDING), body_font))

    if answer.reaction_mhchem and (answer.catalyst or answer.condition):
        parts = []
        if answer.catalyst:
            parts.append(f"Catalyst/Reagent: {answer.catalyst}")
        if answer.condition:
            parts.append(f"Condition: {answer.condition}")
        blocks.append((
            "text",
            wrap_text("   |   ".join(parts), body_font, CANVAS_WIDTH - 2 * PADDING),
            body_font,
        ))

    if answer.node_structures and not answer.mermaid_diagram:
        grid_img = render_structure_grid(answer.node_structures)
        if grid_img:
            blocks.append(("image", grid_img))

    if answer.mermaid_diagram:
        node_smiles = (
            {ns.label: ns.smiles for ns in answer.node_structures}
            if answer.node_structures else None
        )
        flow_img = render_mermaid(answer.mermaid_diagram, node_smiles=node_smiles)
        if flow_img:
            blocks.append(("image", flow_img))
        else:
            blocks.append((
                "text",
                wrap_text(answer.mermaid_diagram, body_font, CANVAS_WIDTH - 2 * PADDING),
                body_font,
            ))

    for table_md in (answer.table_markdown or []):
        table_img = render_table(table_md)
        if table_img:
            blocks.append(("image", table_img))
        else:
            blocks.append(("text", wrap_text(table_md, body_font, CANVAS_WIDTH - 2 * PADDING), body_font))

    for eq in (answer.equation_latex or []):
        eq_img = render_latex(eq)
        if eq_img:
            blocks.append(("image", eq_img))
        else:
            blocks.append(("text", wrap_text(eq, body_font, CANVAS_WIDTH - 2 * PADDING), body_font))

    max_img_width = CANVAS_WIDTH - 2 * PADDING
    scaled_blocks = []
    for block in blocks:
        if block[0] == "image" and block[1].width > max_img_width:
            img = block[1]
            ratio = max_img_width / img.width
            img = img.resize((max_img_width, int(img.height * ratio)))
            scaled_blocks.append(("image", img))
        else:
            scaled_blocks.append(block)
    blocks = scaled_blocks

    line_height = body_font.size + 8
    total_height = PADDING
    for block in blocks:
        if block[0] == "text":
            total_height += len(block[1]) * line_height + 15
        else:
            total_height += block[1].height + 15
    total_height += PADDING

    canvas = Image.new("RGBA", (CANVAS_WIDTH, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    y = PADDING

    for block in blocks:
        if block[0] == "text":
            for line in block[1]:
                draw.text((PADDING, y), line, font=block[2], fill="black")
                y += line_height
            y += 15
        else:
            img = block[1]
            x = (CANVAS_WIDTH - img.width) // 2
            canvas.paste(img, (x, y), img)
            y += img.height + 15

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"answer_{timestamp}.png")
    canvas.convert("RGB").save(out_path)
    return out_path


rendered_path = render_answer(answer)
print(f"Rendered answer saved to: {rendered_path}")

try:
    os.startfile(rendered_path)
except AttributeError:
    pass
