"""
ResumeForge - AI-powered resume analysis, job matching,
optimization, and review system.

Architecture:
    FastAPI
       ↓
    LangGraph
       ↓
    Resume Analyzer
       ↓
    Job Matcher
       ↓
    Resume Optimizer
       ↓
    Resume Reviewer
       ↓
    Conditional Quality Gate
       ├── Re-optimize
       └── Final Result

Supported files:
    PDF / DOCX / TXT

LLM:
    OpenRouter through langchain-openai

Run locally:
    uvicorn server:app --reload

Swagger:
    http://127.0.0.1:8000/docs

Production:
    uvicorn server:app --host 0.0.0.0 --port $PORT
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import json
import logging
import asyncio
from io import BytesIO
from typing import Any, Dict, List, TypedDict

import fitz
import docx

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
    HTTPException,
)
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "").strip()

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing.\n"
        "Add it to your .env file."
    )

if not OPENROUTER_MODEL:
    raise RuntimeError(
        "OPENROUTER_MODEL is missing.\n"
        "Add it to your .env file."
    )


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("resumeforge")


# ============================================================
# APPLICATION CONSTANTS
# ============================================================

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}

MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024
MAX_RESUME_TEXT_CHARS = 25000
MAX_JOB_DESCRIPTION_CHARS = 12000

QUALITY_THRESHOLD = 78
MAX_OPTIMIZATION_ITERATIONS = 2


# ============================================================
# OPENROUTER LLM
# ============================================================

def create_llm(temperature: float = 0.2) -> ChatOpenAI:
    """
    Creates an OpenRouter LLM through its OpenAI-compatible API.

    Configuration comes only from .env:
        OPENROUTER_API_KEY
        OPENROUTER_MODEL
    """

    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=3000,
        max_retries=0,
        timeout=120,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "ResumeForge",
        },
    )


# ============================================================
# LLM RESPONSE HELPERS
# ============================================================

def message_to_text(response: Any) -> str:
    """
    Converts LangChain AIMessage/content blocks into plain text.
    """

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")

                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))

        return "".join(parts).strip()

    return str(content or "").strip()


def extract_json_object(text: str) -> str:
    """
    Extracts the first balanced JSON object from model output.

    Handles:
    - Markdown code fences
    - Explanatory text before JSON
    - Explanatory text after JSON
    """

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    start = text.find("{")

    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):

        char = text[index]

        if in_string:

            if escaped:
                escaped = False

            elif char == "\\":
                escaped = True

            elif char == '"':
                in_string = False

            continue

        if char == '"':
            in_string = True

        elif char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                return text[start:index + 1]

    raise ValueError(
        "JSON object was not properly closed."
    )


def extract_status_code(error: Exception) -> int | None:
    """
    Attempts to extract an HTTP status code from a provider error.
    """

    status_code = getattr(
        error,
        "status_code",
        None,
    )

    if isinstance(status_code, int):
        return status_code

    match = re.search(
        r"(?:status|code|HTTP)[^\d]*(\d{3})",
        str(error),
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return None


def raise_llm_error(error: Exception) -> None:
    """
    Converts common OpenRouter/API failures into useful HTTP errors.
    """

    error_text = str(error).lower()
    status_code = extract_status_code(error)

    if (
        "unauthenticated" in error_text
        or "authentication" in error_text
        or "invalid api key" in error_text
        or "invalid_api_key" in error_text
        or status_code in (401, 403)
    ):
        raise HTTPException(
            status_code=502,
            detail=(
                "OpenRouter authentication failed. "
                "Check OPENROUTER_API_KEY in your .env file."
            ),
        )

    if status_code == 402 or "insufficient credits" in error_text:
        raise HTTPException(
            status_code=502,
            detail=(
                "OpenRouter rejected the request because the selected "
                "model requires credits or the requested output is too "
                "large. ResumeForge now limits each LLM response to "
                "8000 tokens. If this still occurs, set "
                "OPENROUTER_MODEL=openrouter/free in .env."
            ),
        )

    if (
        "quota" in error_text
        or "rate limit" in error_text
        or "ratelimit" in error_text
        or "too many requests" in error_text
        or status_code == 429
    ):
        raise HTTPException(
            status_code=502,
            detail=(
                "OpenRouter rate limit was reached. "
                "Wait briefly and try again."
            ),
        )

    if (
        "model" in error_text
        and (
            "not found" in error_text
            or "does not exist" in error_text
            or "unknown" in error_text
        )
    ):
        raise HTTPException(
            status_code=502,
            detail=(
                f"OpenRouter model '{OPENROUTER_MODEL}' is unavailable. "
                "Check OPENROUTER_MODEL in your .env file."
            ),
        )

    if status_code and status_code >= 500:
        raise HTTPException(
            status_code=502,
            detail=(
                "OpenRouter temporarily returned a server error. "
                "Please try again."
            ),
        )

    # Preserve useful provider errors instead of hiding everything.
    raise HTTPException(
        status_code=502,
        detail=f"OpenRouter request failed: {str(error)}",
    )


# ============================================================
# STRUCTURED OPENROUTER INVOCATION
# ============================================================

def invoke_structured(
    prompt: str,
    schema_model: type[BaseModel],
    system_prompt: str,
) -> BaseModel:
    """
    Generates structured Pydantic output through OpenRouter.

    Strategy:
    1. Try LangChain's native JSON-schema structured output.
    2. If the selected model/provider does not support it, fall back
       to a strict JSON prompt and validate the response with Pydantic.
    """

    llm = create_llm()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]

    # --------------------------------------------------------
    # METHOD 1: Native structured output
    # --------------------------------------------------------

    try:
        structured_llm = llm.with_structured_output(
            schema_model,
            method="json_schema",
        )

        result = structured_llm.invoke(messages)

        if isinstance(result, schema_model):
            return result

        if isinstance(result, dict):
            return schema_model.model_validate(result)

        raise ValueError(
            f"Unexpected structured output type: {type(result)}"
        )

    except HTTPException:
        raise

    except Exception as native_error:
        logger.warning(
            "Native OpenRouter structured output failed. "
            "Using JSON fallback. Error: %s",
            native_error,
        )

        # Authentication, quota, model-access and server errors should
        # not be hidden by the JSON fallback.
        status_code = extract_status_code(native_error)
        error_text = str(native_error).lower()

        fatal_provider_error = (
            status_code in (401, 402, 403, 429)
            or "invalid api key" in error_text
            or "authentication" in error_text
            or "unauthenticated" in error_text
            or "quota" in error_text
            or "rate limit" in error_text
            or "ratelimit" in error_text
            or (
                "model" in error_text
                and (
                    "not found" in error_text
                    or "does not exist" in error_text
                )
            )
        )

        if fatal_provider_error:
            raise_llm_error(native_error)

    # --------------------------------------------------------
    # METHOD 2: JSON fallback
    # --------------------------------------------------------

    schema = schema_model.model_json_schema()

    fallback_system = (
        system_prompt
        + "\n\n"
        + "OUTPUT RULES:\n"
        + "Return ONLY one valid JSON object.\n"
        + "Do not use markdown.\n"
        + "Do not add explanations.\n"
        + "Do not add text before or after the JSON.\n"
        + "The JSON must follow this schema exactly:\n"
        + json.dumps(schema)
    )

    fallback_messages = [
        SystemMessage(content=fallback_system),
        HumanMessage(content=prompt),
    ]

    last_error = None

    for attempt in range(2):
        try:
            raw_response = llm.invoke(fallback_messages)

            text = message_to_text(raw_response)
            json_text = extract_json_object(text)
            parsed = json.loads(json_text)

            return schema_model.model_validate(parsed)

        except HTTPException:
            raise

        except Exception as fallback_error:
            status_code = extract_status_code(fallback_error)
            error_text = str(fallback_error).lower()

            if (
                status_code in (401, 402, 403, 429)
                or "invalid api key" in error_text
                or "authentication" in error_text
                or "unauthenticated" in error_text
                or "quota" in error_text
                or "rate limit" in error_text
                or "ratelimit" in error_text
            ):
                raise_llm_error(fallback_error)

            last_error = fallback_error

            logger.warning(
                "JSON fallback attempt %s failed: %s",
                attempt + 1,
                fallback_error,
            )

            fallback_messages = [
                SystemMessage(
                    content=(
                        fallback_system
                        + "\n\n"
                        + "IMPORTANT:\n"
                        + "Your previous response was invalid.\n"
                        + "Return ONLY raw valid JSON."
                    )
                ),
                HumanMessage(content=prompt),
            ]

    raise HTTPException(
        status_code=502,
        detail=(
            "OpenRouter returned an invalid structured response. "
            "Try again or choose a model with reliable JSON output. "
            f"Internal parsing error: {last_error}"
        ),
    )


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf(content: bytes) -> str:

    try:
        document = fitz.open(
            stream=content,
            filetype="pdf",
        )

    except Exception as error:

        logger.error(
            "PDF opening failed: %s",
            error,
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not open the PDF. "
                "The file may be corrupted or password protected."
            ),
        )

    try:

        pages = []

        for page in document:
            text = page.get_text()

            if text:
                pages.append(text)

        result = "\n".join(pages).strip()

    finally:

        document.close()

    if not result:
        raise HTTPException(
            status_code=400,
            detail=(
                "No text could be extracted from this PDF. "
                "If it is a scanned resume, OCR is required."
            ),
        )

    return result


def extract_docx(content: bytes) -> str:

    try:

        document = docx.Document(
            BytesIO(content)
        )

    except Exception as error:

        logger.error(
            "DOCX opening failed: %s",
            error,
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not open the DOCX file. "
                "The document may be corrupted."
            ),
        )

    sections = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            sections.append(text)

    for table in document.tables:

        for row in table.rows:

            row_values = []

            for cell in row.cells:

                cell_text = cell.text.strip()

                if cell_text:
                    row_values.append(cell_text)

            if row_values:
                sections.append(
                    " | ".join(row_values)
                )

    result = "\n".join(sections).strip()

    if not result:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in the DOCX file.",
        )

    return result


def extract_txt(content: bytes) -> str:

    try:

        result = content.decode(
            "utf-8"
        ).strip()

    except UnicodeDecodeError:

        try:

            result = content.decode(
                "latin-1"
            ).strip()

        except Exception:

            raise HTTPException(
                status_code=400,
                detail="Could not decode the TXT file.",
            )

    if not result:
        raise HTTPException(
            status_code=400,
            detail="The TXT file is empty.",
        )

    return result


def extract_resume_text(
    filename: str,
    content: bytes,
) -> str:

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Resume filename is missing.",
        )

    extension = os.path.splitext(
        filename.lower()
    )[1]

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension or 'unknown'}. "
                "Allowed: PDF, DOCX, TXT."
            ),
        )

    if not content:

        raise HTTPException(
            status_code=400,
            detail="Uploaded resume is empty.",
        )

    if len(content) > MAX_FILE_SIZE_BYTES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Resume is too large. "
                "Maximum allowed size is 8 MB."
            ),
        )

    if extension == ".pdf":
        text = extract_pdf(content)

    elif extension == ".docx":
        text = extract_docx(content)

    else:
        text = extract_txt(content)

    text = text.strip()

    if len(text) > MAX_RESUME_TEXT_CHARS:

        logger.warning(
            "Resume text exceeded limit and was truncated."
        )

        text = text[
            :MAX_RESUME_TEXT_CHARS
        ]

    return text


# ============================================================
# PYDANTIC MODELS
# ============================================================

class ResumeAnalysis(BaseModel):

    contact_information: str = ""

    summary: str = ""

    skills: List[str] = Field(
        default_factory=list
    )

    technical_skills: List[str] = Field(
        default_factory=list
    )

    soft_skills: List[str] = Field(
        default_factory=list
    )

    education: List[str] = Field(
        default_factory=list
    )

    experience: List[str] = Field(
        default_factory=list
    )

    projects: List[str] = Field(
        default_factory=list
    )

    certifications: List[str] = Field(
        default_factory=list
    )

    achievements: List[str] = Field(
        default_factory=list
    )

    technologies: List[str] = Field(
        default_factory=list
    )

    quantified_achievements: List[str] = Field(
        default_factory=list
    )

    weak_sections: List[str] = Field(
        default_factory=list
    )

    missing_information: List[str] = Field(
        default_factory=list
    )


class JobMatch(BaseModel):

    matching_skills: List[str] = Field(
        default_factory=list
    )

    missing_skills: List[str] = Field(
        default_factory=list
    )

    partially_matching_skills: List[str] = Field(
        default_factory=list
    )

    matching_experience: List[str] = Field(
        default_factory=list
    )

    missing_experience: List[str] = Field(
        default_factory=list
    )

    matching_keywords: List[str] = Field(
        default_factory=list
    )

    missing_keywords: List[str] = Field(
        default_factory=list
    )

    role_alignment: str = ""

    experience_alignment: str = ""

    overall_job_fit: str = ""

    fit_score: int = Field(
        default=0,
        ge=0,
        le=100,
    )


class OptimizedResume(BaseModel):

    contact: str = ""

    professional_summary: str = ""

    skills: List[str] = Field(
        default_factory=list
    )

    experience: List[str] = Field(
        default_factory=list
    )

    projects: List[str] = Field(
        default_factory=list
    )

    education: List[str] = Field(
        default_factory=list
    )

    certifications: List[str] = Field(
        default_factory=list
    )

    achievements: List[str] = Field(
        default_factory=list
    )

    keyword_improvements: List[str] = Field(
        default_factory=list
    )


class ResumeReview(BaseModel):

    ats_relevance: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    keyword_coverage: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    role_relevance: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    clarity: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    grammar: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    consistency: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    achievement_focus: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    technical_relevance: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    professional_quality: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    factual_consistency: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    overall_quality: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    remaining_issues: List[str] = Field(
        default_factory=list
    )

    remaining_missing_keywords: List[str] = Field(
        default_factory=list
    )

    potential_hallucinations: List[str] = Field(
        default_factory=list
    )

    weak_bullets: List[str] = Field(
        default_factory=list
    )

    recommendations: List[str] = Field(
        default_factory=list
    )


# ============================================================
# LANGGRAPH STATE
# ============================================================

class ResumeState(TypedDict, total=False):

    resume_filename: str

    resume_text: str

    original_resume: str

    target_role: str

    job_description: str

    resume_analysis: Dict[str, Any]

    job_match: Dict[str, Any]

    optimized_resume: Dict[str, Any]

    review_result: Dict[str, Any]

    optimization_iteration: int

    max_iterations: int


# ============================================================
# AGENT 1 — RESUME ANALYZER
# ============================================================

def analyzer_node(
    state: ResumeState,
) -> Dict[str, Any]:

    logger.info(
        "Resume Analyzer started."
    )

    system_prompt = """
You are ResumeForge's Resume Analyzer Agent.

Your job is to extract factual information from the
candidate's resume.

STRICT RULES:

1. Use ONLY information explicitly present in the resume.
2. Never invent information.
3. Never infer employment or skills without evidence.
4. Keep missing information separate.
5. Preserve technologies exactly when possible.
6. Identify weak sections honestly.
7. Identify quantified achievements only when the
   original resume actually contains numbers or metrics.

Return the requested structured information.
"""

    prompt = f"""
TARGET ROLE:
{state["target_role"]}

RESUME:

{state["resume_text"]}

Analyze the resume.
"""

    result = invoke_structured(
        prompt,
        ResumeAnalysis,
        system_prompt,
    )

    logger.info(
        "Resume Analyzer completed."
    )

    return {
        "resume_analysis": result.model_dump()
    }


# ============================================================
# AGENT 2 — JOB MATCHER
# ============================================================

def matcher_node(
    state: ResumeState,
) -> Dict[str, Any]:

    logger.info(
        "Job Matcher started."
    )

    system_prompt = """
You are ResumeForge's Job Matcher Agent.

Compare the candidate's resume against the target
job description.

IMPORTANT:

A missing skill from the resume does NOT prove that
the candidate does not possess that skill.

Use wording such as:

"Not demonstrated in the provided resume."

Do not make unsupported claims.

The fit score must reflect actual evidence in the
resume compared with the job description.

Be strict and realistic.
"""

    prompt = f"""
TARGET ROLE:
{state["target_role"]}

JOB DESCRIPTION:

{state["job_description"]}

RESUME ANALYSIS:

{json.dumps(
    state["resume_analysis"],
    indent=2
)}

Determine:

- matching skills
- missing skills
- partially matching skills
- matching experience
- missing experience
- matching keywords
- missing keywords
- role alignment
- experience alignment
- overall fit
- fit score from 0 to 100
"""

    result = invoke_structured(
        prompt,
        JobMatch,
        system_prompt,
    )

    logger.info(
        "Job Matcher completed. Score=%s",
        result.fit_score,
    )

    return {
        "job_match": result.model_dump()
    }


# ============================================================
# AGENT 3 — RESUME OPTIMIZER
# ============================================================

def optimizer_node(
    state: ResumeState,
) -> Dict[str, Any]:

    iteration = (
        state.get(
            "optimization_iteration",
            0,
        )
        + 1
    )

    logger.info(
        "Resume Optimizer started. Iteration=%s",
        iteration,
    )

    previous_feedback = ""

    previous_review = state.get(
        "review_result"
    )

    if previous_review:

        previous_feedback = f"""
PREVIOUS REVIEW FEEDBACK:

Remaining issues:
{json.dumps(
    previous_review.get(
        "remaining_issues",
        []
    )
)}

Missing keywords:
{json.dumps(
    previous_review.get(
        "remaining_missing_keywords",
        []
    )
)}

Potential hallucinations:
{json.dumps(
    previous_review.get(
        "potential_hallucinations",
        []
    )
)}

Weak bullets:
{json.dumps(
    previous_review.get(
        "weak_bullets",
        []
    )
)}

Fix these issues in this revision.
"""

    system_prompt = """
You are ResumeForge's Resume Optimizer Agent.

Your task is to improve the candidate's resume
for the target role and job description.

You may:

- improve wording
- improve action verbs
- improve clarity
- improve bullet structure
- improve achievement focus
- improve keyword placement
- improve professional summary
- reorder information
- emphasize skills already demonstrated
- improve ATS relevance

ABSOLUTE FACTUAL RULE:

Every factual claim must be supported by the
original resume.

NEVER INVENT:

- companies
- job titles
- technologies
- projects
- certifications
- degrees
- responsibilities
- achievements
- numbers
- percentages
- users
- revenue
- rankings
- awards
- dates
- employment history

If a metric does not exist in the original resume,
DO NOT create one.

For example:

Original:
"Built an automation system using Python."

Allowed:
"Built a Python-based automation system to
streamline repetitive workflow tasks."

Forbidden:
"Built a Python automation system that reduced
processing time by 60%."

Do not add unsupported skills just because the
job description mentions them.

If a keyword is missing but not supported by the
resume, do not insert it as if the candidate has it.

The optimized resume must remain truthful.
"""

    prompt = f"""
TARGET ROLE:

{state["target_role"]}

JOB DESCRIPTION:

{state["job_description"]}

ORIGINAL RESUME:

{state["original_resume"]}

RESUME ANALYSIS:

{json.dumps(
    state["resume_analysis"],
    indent=2
)}

JOB MATCH:

{json.dumps(
    state["job_match"],
    indent=2
)}

{previous_feedback}

Create an improved resume using structured sections.

Do not invent information.
"""

    result = invoke_structured(
        prompt,
        OptimizedResume,
        system_prompt,
    )

    logger.info(
        "Resume Optimizer completed."
    )

    return {
        "optimized_resume": result.model_dump(),
        "optimization_iteration": iteration,
    }


# ============================================================
# AGENT 4 — RESUME REVIEWER
# ============================================================

def reviewer_node(
    state: ResumeState,
) -> Dict[str, Any]:

    logger.info(
        "Resume Reviewer started."
    )

    system_prompt = """
You are ResumeForge's final Resume Reviewer Agent.

You are a strict ATS and professional resume reviewer.

Compare:

1. Original resume
2. Optimized resume
3. Job description

Evaluate:

- ATS relevance
- keyword coverage
- role relevance
- clarity
- grammar
- consistency
- achievement focus
- technical relevance
- professional quality
- factual consistency

MOST IMPORTANT:

Check whether the optimized resume introduced
ANY information unsupported by the original resume.

Flag:

- invented metrics
- invented technologies
- invented projects
- invented companies
- invented job titles
- invented achievements
- invented dates
- invented responsibilities

Do not inflate scores.

If something is unsupported, list it under
potential_hallucinations.
"""

    prompt = f"""
TARGET ROLE:

{state["target_role"]}

JOB DESCRIPTION:

{state["job_description"]}

ORIGINAL RESUME:

{state["original_resume"]}

OPTIMIZED RESUME:

{json.dumps(
    state["optimized_resume"],
    indent=2
)}

RESUME ANALYSIS:

{json.dumps(
    state["resume_analysis"],
    indent=2
)}

JOB MATCH:

{json.dumps(
    state["job_match"],
    indent=2
)}

Review the optimized resume.
"""

    result = invoke_structured(
        prompt,
        ResumeReview,
        system_prompt,
    )

    logger.info(
        "Resume Reviewer completed. Quality=%s",
        result.overall_quality,
    )

    return {
        "review_result": result.model_dump()
    }


# ============================================================
# LANGGRAPH CONDITIONAL ROUTING
# ============================================================

def route_after_review(
    state: ResumeState,
) -> str:

    review = state.get(
        "review_result",
        {},
    )

    quality = review.get(
        "overall_quality",
        0,
    )

    hallucinations = review.get(
        "potential_hallucinations",
        [],
    )

    iteration = state.get(
        "optimization_iteration",
        0,
    )

    max_iterations = state.get(
        "max_iterations",
        MAX_OPTIMIZATION_ITERATIONS,
    )

    # --------------------------------------------------------
    # Maximum iterations reached
    # --------------------------------------------------------

    if iteration >= max_iterations:

        logger.info(
            "Maximum iterations reached."
        )

        return "finish"

    # --------------------------------------------------------
    # Quality accepted
    # --------------------------------------------------------

    if (
        quality >= QUALITY_THRESHOLD
        and not hallucinations
    ):

        logger.info(
            "Resume accepted. Quality=%s",
            quality,
        )

        return "finish"

    # --------------------------------------------------------
    # Re-optimize
    # --------------------------------------------------------

    logger.info(
        "Resume requires another optimization pass."
    )

    return "optimize"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(
    ResumeState
)

workflow.add_node(
    "analyzer",
    analyzer_node,
)

workflow.add_node(
    "matcher",
    matcher_node,
)

workflow.add_node(
    "optimizer",
    optimizer_node,
)

workflow.add_node(
    "reviewer",
    reviewer_node,
)

workflow.add_edge(
    START,
    "analyzer",
)

workflow.add_edge(
    "analyzer",
    "matcher",
)

workflow.add_edge(
    "matcher",
    "optimizer",
)

workflow.add_edge(
    "optimizer",
    "reviewer",
)

workflow.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "optimize": "optimizer",
        "finish": END,
    },
)

graph = workflow.compile()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ResumeForge",
    description=(
        "AI-powered resume analysis, job matching, "
        "optimization, and review system."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ============================================================
# EMBEDDED FRONTEND
# ============================================================

FRONTEND_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ResumeForge — AI Resume Intelligence</title>
<style>
:root{
    --bg:#0a0f1c;
    --panel:#111827;
    --panel2:#0f172a;
    --border:#243044;
    --text:#f8fafc;
    --muted:#94a3b8;
    --accent:#7c3aed;
    --accent2:#2563eb;
    --success:#22c55e;
    --warning:#f59e0b;
    --danger:#ef4444;
}
*{box-sizing:border-box}
body{
    margin:0;
    font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    background:
      radial-gradient(circle at 15% 10%,rgba(124,58,237,.18),transparent 30%),
      radial-gradient(circle at 85% 20%,rgba(37,99,235,.14),transparent 28%),
      var(--bg);
    color:var(--text);
    min-height:100vh;
}
.container{width:min(1180px,92%);margin:auto}
header{padding:42px 0 24px}
.brand{display:flex;align-items:center;gap:14px}
.logo{
    width:48px;height:48px;border-radius:14px;
    display:grid;place-items:center;
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    font-size:24px;font-weight:800;
}
h1{font-size:34px;margin:0}
.subtitle{color:var(--muted);margin:8px 0 0}
.card{
    background:rgba(17,24,39,.88);
    border:1px solid var(--border);
    border-radius:20px;
    padding:24px;
    box-shadow:0 18px 50px rgba(0,0,0,.22);
    margin-bottom:20px;
}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:800px){.grid{grid-template-columns:1fr}}
label{display:block;font-weight:700;margin-bottom:9px}
input,textarea{
    width:100%;border:1px solid var(--border);
    background:#0b1220;color:var(--text);
    border-radius:12px;padding:13px 14px;
    outline:none;font:inherit;
}
textarea{min-height:180px;resize:vertical}
input:focus,textarea:focus{border-color:var(--accent)}
.drop{
    border:2px dashed #334155;
    border-radius:16px;padding:30px 20px;text-align:center;
    cursor:pointer;background:#0b1220;
    transition:.2s;
}
.drop:hover,.drop.drag{border-color:var(--accent);background:#11152a}
.drop strong{display:block;font-size:18px;margin-bottom:7px}
.drop span{color:var(--muted);font-size:14px}
input[type=file]{display:none}
.file-name{margin-top:12px;color:#cbd5e1;font-size:14px}
.btn{
    width:100%;border:0;border-radius:13px;padding:15px;
    color:white;background:linear-gradient(135deg,var(--accent),var(--accent2));
    font-size:16px;font-weight:800;cursor:pointer;
    margin-top:20px;
}
.btn:disabled{opacity:.5;cursor:not-allowed}
.status{
    display:none;margin-top:18px;padding:16px;
    border-radius:14px;background:#0b1220;border:1px solid var(--border);
}
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px}
@media(max-width:700px){.steps{grid-template-columns:1fr 1fr}}
.step{padding:10px;border-radius:10px;background:#111827;color:#64748b;font-size:13px;text-align:center}
.step.active{color:white;border:1px solid var(--accent);background:#1b1640}
.step.done{color:#86efac;border:1px solid #166534;background:#0c2015}
.error{
    display:none;padding:14px;border-radius:12px;
    background:#2a1115;border:1px solid #7f1d1d;color:#fecaca;margin-top:15px;
}
.results{display:none}
.hero-score{display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap}
.score{
    width:130px;height:130px;border-radius:50%;
    display:grid;place-items:center;
    background:conic-gradient(var(--accent) 0deg,#1e293b 0deg);
    position:relative;
}
.score:before{
    content:"";position:absolute;inset:9px;border-radius:50%;background:var(--panel);
}
.score b{position:relative;font-size:28px}
.score small{position:absolute;transform:translateY(30px);color:var(--muted)}
.section-title{font-size:20px;margin:0 0 14px}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{padding:7px 10px;border-radius:999px;background:#1e293b;color:#dbeafe;font-size:13px}
.chip.good{background:#0b2a19;color:#86efac}
.chip.warn{background:#2a1d08;color:#fcd34d}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:800px){.two{grid-template-columns:1fr}}
.metric-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
@media(max-width:900px){.metric-grid{grid-template-columns:repeat(2,1fr)}}
.metric{background:#0b1220;border:1px solid var(--border);border-radius:12px;padding:14px}
.metric span{display:block;color:var(--muted);font-size:12px}
.metric b{font-size:22px}
.resume-section{
    background:#0b1220;border:1px solid var(--border);
    border-radius:14px;padding:16px;margin-top:12px;
}
.resume-section h3{margin:0 0 10px;font-size:15px;color:#c4b5fd}
.resume-section p{white-space:pre-wrap;line-height:1.65;margin:0;color:#e2e8f0}
.resume-section ul{margin:0;padding-left:20px;color:#e2e8f0}
.resume-section li{margin:6px 0;line-height:1.5}
.loading{display:none;text-align:center;padding:40px;color:var(--muted)}
.spinner{
    width:38px;height:38px;border:4px solid #334155;
    border-top-color:var(--accent);border-radius:50%;
    animation:spin 1s linear infinite;margin:0 auto 14px;
}
@keyframes spin{to{transform:rotate(360deg)}}
a{color:#c4b5fd;text-decoration:none}
footer{padding:10px 0 40px;color:#64748b;text-align:center;font-size:13px}
.badge{
    display:inline-block;padding:5px 9px;border-radius:999px;
    background:#1e1b4b;color:#c4b5fd;font-size:12px;margin-bottom:12px;
}
.empty{color:var(--muted);font-size:14px}
</style>
</head>
<body>
<div class="container">
<header>
    <div class="brand">
        <div class="logo">RF</div>
        <div>
            <h1>ResumeForge</h1>
            <p class="subtitle">AI Resume Intelligence & Optimization</p>
        </div>
    </div>
</header>

<section class="card">
    <span class="badge">LangGraph Multi-Agent System</span>
    <h2>Analyze. Match. Optimize. Review.</h2>
    <p class="subtitle">
        Upload your resume, provide the target role and job description,
        and ResumeForge will build a truthful, job-focused improvement report.
    </p>

    <div class="grid" style="margin-top:22px">
        <div>
            <label>Resume</label>
            <label class="drop" id="dropZone" for="resumeFile">
                <strong>📄 Drop your resume here</strong>
                <span>PDF, DOCX or TXT — maximum 8 MB</span>
                <input id="resumeFile" type="file" accept=".pdf,.docx,.txt">
                <div class="file-name" id="fileName">No file selected</div>
            </label>
        </div>

        <div>
            <label for="role">Target Job Role</label>
            <input id="role" placeholder="e.g. AI Engineer">

            <label for="jd" style="margin-top:18px">Job Description</label>
            <textarea id="jd" placeholder="Paste the complete job description here..."></textarea>
        </div>
    </div>

    <button class="btn" id="analyzeBtn">🚀 Analyze My Resume</button>

    <div class="status" id="status">
        <strong id="statusText">Starting ResumeForge...</strong>
        <div class="steps">
            <div class="step" id="s1">Resume Analyzer</div>
            <div class="step" id="s2">Job Matcher</div>
            <div class="step" id="s3">Resume Optimizer</div>
            <div class="step" id="s4">Resume Reviewer</div>
        </div>
    </div>

    <div class="error" id="error"></div>
</section>

<div class="loading" id="loading">
    <div class="spinner"></div>
    <div>ResumeForge is running the AI workflow...</div>
</div>

<section class="results" id="results">

    <div class="card hero-score">
        <div>
            <span class="badge">Job Match</span>
            <h2 id="roleResult">Target Role</h2>
            <p class="subtitle">Evidence-based fit against the supplied job description.</p>
        </div>
        <div class="score" id="scoreCircle">
            <b id="scoreValue">0</b>
            <small>/100</small>
        </div>
    </div>

    <div class="card">
        <h2 class="section-title">🎯 Job Match Insights</h2>
        <div class="two">
            <div>
                <h3>Matching Skills</h3>
                <div class="chips" id="matchingSkills"></div>
            </div>
            <div>
                <h3>Not Demonstrated / Missing</h3>
                <div class="chips" id="missingSkills"></div>
            </div>
        </div>
        <div class="two" style="margin-top:20px">
            <div>
                <h3>Matching Keywords</h3>
                <div class="chips" id="matchingKeywords"></div>
            </div>
            <div>
                <h3>Missing Keywords</h3>
                <div class="chips" id="missingKeywords"></div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2 class="section-title">📊 Candidate Analysis</h2>
        <div class="two">
            <div class="resume-section">
                <h3>Technical Skills</h3>
                <div class="chips" id="technicalSkills"></div>
            </div>
            <div class="resume-section">
                <h3>Technologies</h3>
                <div class="chips" id="technologies"></div>
            </div>
        </div>
        <div class="two">
            <div class="resume-section">
                <h3>Weak Sections</h3>
                <ul id="weakSections"></ul>
            </div>
            <div class="resume-section">
                <h3>Missing Information</h3>
                <ul id="missingInformation"></ul>
            </div>
        </div>
    </div>

    <div class="card">
        <h2 class="section-title">✨ Optimized Resume</h2>
        <div id="optimizedResume"></div>
    </div>

    <div class="card">
        <h2 class="section-title">🔍 Reviewer Scores</h2>
        <div class="metric-grid" id="reviewMetrics"></div>

        <div class="two" style="margin-top:18px">
            <div class="resume-section">
                <h3>Remaining Issues</h3>
                <ul id="remainingIssues"></ul>
            </div>
            <div class="resume-section">
                <h3>Recommendations</h3>
                <ul id="recommendations"></ul>
            </div>
        </div>

        <div class="resume-section">
            <h3>Hallucination / Factual Consistency Check</h3>
            <div id="hallucinations"></div>
        </div>
    </div>

</section>

<footer>
    ResumeForge • OpenRouter + LangChain + LangGraph + FastAPI
    • <a href="/docs" target="_blank">Open API Docs</a>
</footer>
</div>

<script>
const $ = id => document.getElementById(id);

const fileInput = $("resumeFile");
const dropZone = $("dropZone");

fileInput.addEventListener("change", () => {
    $("fileName").textContent =
        fileInput.files.length ? fileInput.files[0].name : "No file selected";
});

["dragenter","dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
        e.preventDefault();
        dropZone.classList.add("drag");
    });
});

["dragleave","drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
        e.preventDefault();
        dropZone.classList.remove("drag");
    });
});

dropZone.addEventListener("drop", e => {
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        $("fileName").textContent = e.dataTransfer.files[0].name;
    }
});

function setStep(index, state) {
    const el = $("s" + index);
    el.className = "step " + state;
}

function resetSteps() {
    for(let i=1;i<=4;i++) setStep(i, "");
}

function setProgress(index, message) {
    resetSteps();
    for(let i=1;i<index;i++) setStep(i, "done");
    if(index <= 4) setStep(index, "active");
    $("statusText").textContent = message;
}

function showError(message) {
    $("error").style.display = "block";
    $("error").textContent = message;
}

function hideError() {
    $("error").style.display = "none";
}

function chips(id, items, type="") {
    const el = $(id);
    el.innerHTML = "";
    if(!items || !items.length) {
        el.innerHTML = '<span class="empty">None identified.</span>';
        return;
    }
    items.forEach(item => {
        const span = document.createElement("span");
        span.className = "chip " + type;
        span.textContent = item;
        el.appendChild(span);
    });
}

function list(id, items) {
    const el = $(id);
    el.innerHTML = "";
    if(!items || !items.length) {
        el.innerHTML = '<li class="empty">None identified.</li>';
        return;
    }
    items.forEach(item => {
        const li = document.createElement("li");
        li.textContent = item;
        el.appendChild(li);
    });
}

function addResumeSection(container, title, value) {
    const box = document.createElement("div");
    box.className = "resume-section";

    const h = document.createElement("h3");
    h.textContent = title;
    box.appendChild(h);

    if(Array.isArray(value)) {
        const ul = document.createElement("ul");
        value.forEach(item => {
            const li = document.createElement("li");
            li.textContent = item;
            ul.appendChild(li);
        });
        if(!value.length) {
            const p = document.createElement("p");
            p.className = "empty";
            p.textContent = "No information available.";
            box.appendChild(p);
        } else {
            box.appendChild(ul);
        }
    } else {
        const p = document.createElement("p");
        p.textContent = value || "No information available.";
        box.appendChild(p);
    }

    $(container).appendChild(box);
}

function renderResults(data) {
    $("results").style.display = "block";
    $("roleResult").textContent = data.target_role || "Target Role";

    const score = Number(data.job_fit_score || 0);
    $("scoreValue").textContent = score;
    $("scoreCircle").style.background =
        `conic-gradient(var(--accent) ${score * 3.6}deg, #1e293b 0deg)`;

    const match = data.job_match || {};
    const analysis = data.analysis || {};
    const optimized = data.optimized_resume || {};
    const review = data.review || {};

    chips("matchingSkills", match.matching_skills || [], "good");
    chips("missingSkills", match.missing_skills || [], "warn");
    chips("matchingKeywords", match.matching_keywords || [], "good");
    chips("missingKeywords", match.missing_keywords || [], "warn");

    chips("technicalSkills", analysis.technical_skills || []);
    chips("technologies", analysis.technologies || []);

    list("weakSections", analysis.weak_sections || []);
    list("missingInformation", analysis.missing_information || []);

    const resume = $("optimizedResume");
    resume.innerHTML = "";

    addResumeSection(resume, "Professional Summary", optimized.professional_summary);
    addResumeSection(resume, "Skills", optimized.skills);
    addResumeSection(resume, "Experience", optimized.experience);
    addResumeSection(resume, "Projects", optimized.projects);
    addResumeSection(resume, "Education", optimized.education);
    addResumeSection(resume, "Certifications", optimized.certifications);
    addResumeSection(resume, "Achievements", optimized.achievements);

    const metricData = [
        ["ATS Relevance", review.ats_relevance],
        ["Keyword Coverage", review.keyword_coverage],
        ["Role Relevance", review.role_relevance],
        ["Clarity", review.clarity],
        ["Grammar", review.grammar],
        ["Consistency", review.consistency],
        ["Achievement Focus", review.achievement_focus],
        ["Technical Relevance", review.technical_relevance],
        ["Professional Quality", review.professional_quality],
        ["Factual Consistency", review.factual_consistency],
        ["Overall Quality", review.overall_quality]
    ];

    $("reviewMetrics").innerHTML = "";
    metricData.forEach(([name, value]) => {
        const div = document.createElement("div");
        div.className = "metric";
        div.innerHTML = `<span>${name}</span><b>${Number(value || 0)}/100</b>`;
        $("reviewMetrics").appendChild(div);
    });

    list("remainingIssues", review.remaining_issues || []);
    list("recommendations", review.recommendations || []);

    const hallucinations = review.potential_hallucinations || [];
    $("hallucinations").innerHTML = "";

    if(hallucinations.length) {
        hallucinations.forEach(item => {
            const p = document.createElement("p");
            p.style.color = "#fca5a5";
            p.textContent = "⚠ " + item;
            $("hallucinations").appendChild(p);
        });
    } else {
        const p = document.createElement("p");
        p.style.color = "#86efac";
        p.textContent = "✓ No unsupported factual additions were detected.";
        $("hallucinations").appendChild(p);
    }
}

$("analyzeBtn").addEventListener("click", async () => {
    hideError();

    const file = fileInput.files[0];
    const role = $("role").value.trim();
    const jd = $("jd").value.trim();

    if(!file) {
        showError("Please upload a PDF, DOCX, or TXT resume.");
        return;
    }

    if(!role) {
        showError("Please enter the target job role.");
        return;
    }

    if(!jd) {
        showError("Please paste the job description.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_role", role);
    formData.append("job_description", jd);

    $("analyzeBtn").disabled = true;
    $("status").style.display = "block";
    $("loading").style.display = "block";
    $("results").style.display = "none";
    resetSteps();

    try {
        setProgress(1, "Resume Analyzer is extracting and analyzing your resume...");

        // The backend runs the whole graph as one request. These stages
        // are UI progress hints; the backend does not stream node events.
        setTimeout(() => setProgress(2, "Job Matcher is comparing your resume with the job description..."), 1200);
        setTimeout(() => setProgress(3, "Resume Optimizer is improving relevant sections..."), 3000);
        setTimeout(() => setProgress(4, "Resume Reviewer is validating the optimized resume..."), 5000);

        const response = await fetch("/resume/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if(!response.ok) {
            throw new Error(data.detail || "Resume analysis failed.");
        }

        for(let i=1;i<=4;i++) setStep(i, "done");
        $("statusText").textContent =
            `Analysis complete • ${data.optimization_iterations || 1} optimization pass(es)`;

        renderResults(data);

        $("results").scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    } catch(error) {
        showError(error.message || "Something went wrong.");
        resetSteps();
    } finally {
        $("analyzeBtn").disabled = false;
        $("loading").style.display = "none";
    }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def frontend():
    """Serve the embedded ResumeForge frontend."""
    return HTMLResponse(content=FRONTEND_HTML)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "service": "ResumeForge",
    }


# ============================================================
# EXTRACT ENDPOINT
# ============================================================

@app.post("/resume/extract")
async def extract_resume(
    file: UploadFile = File(...),
):

    content = await file.read()

    text = extract_resume_text(
        file.filename or "",
        content,
    )

    return {
        "status": "success",
        "filename": file.filename,
        "character_count": len(text),
        "extracted_text": text,
    }


# ============================================================
# MAIN RESUME ANALYSIS ENDPOINT
# ============================================================

@app.post("/resume/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    target_role: str = Form(...),
    job_description: str = Form(...),
):

    # --------------------------------------------------------
    # Validate target role
    # --------------------------------------------------------

    target_role = target_role.strip()

    if not target_role:

        raise HTTPException(
            status_code=400,
            detail="target_role cannot be empty.",
        )

    # --------------------------------------------------------
    # Validate job description
    # --------------------------------------------------------

    job_description = job_description.strip()

    if not job_description:

        raise HTTPException(
            status_code=400,
            detail="job_description cannot be empty.",
        )

    if len(job_description) > MAX_JOB_DESCRIPTION_CHARS:

        job_description = (
            job_description[
                :MAX_JOB_DESCRIPTION_CHARS
            ]
        )

        logger.warning(
            "Job description truncated."
        )

    # --------------------------------------------------------
    # Read uploaded resume
    # --------------------------------------------------------

    content = await file.read()

    resume_text = extract_resume_text(
        file.filename or "",
        content,
    )

    logger.info(
        "Resume extracted successfully. File=%s",
        file.filename,
    )

    # --------------------------------------------------------
    # Initial LangGraph state
    # --------------------------------------------------------

    initial_state: ResumeState = {

        "resume_filename": (
            file.filename
            or "resume"
        ),

        "resume_text": resume_text,

        "original_resume": resume_text,

        "target_role": target_role,

        "job_description": job_description,

        "optimization_iteration": 0,

        "max_iterations": (
            MAX_OPTIMIZATION_ITERATIONS
        ),
    }

    # --------------------------------------------------------
    # Run LangGraph outside FastAPI event loop
    # --------------------------------------------------------

    try:

        final_state = await asyncio.to_thread(
            graph.invoke,
            initial_state,
        )

    except HTTPException:
        raise

    except Exception as error:

        logger.exception(
            "LangGraph workflow failed."
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "ResumeForge AI workflow failed. "
                "Check the server logs and verify "
                "your OpenRouter API key/model."
            ),
        ) from error

    # --------------------------------------------------------
    # Extract final results
    # --------------------------------------------------------

    analysis = final_state.get(
        "resume_analysis",
        {},
    )

    job_match = final_state.get(
        "job_match",
        {},
    )

    optimized_resume = final_state.get(
        "optimized_resume",
        {},
    )

    review = final_state.get(
        "review_result",
        {},
    )

    return {
        "status": "success",

        "target_role": target_role,

        "job_fit_score": job_match.get(
            "fit_score",
            0,
        ),

        "analysis": analysis,

        "job_match": job_match,

        "optimized_resume": optimized_resume,

        "review": review,

        "optimization_iterations": final_state.get(
            "optimization_iteration",
            0,
        ),

        "final_recommendations": review.get(
            "recommendations",
            [],
        ),
    }


# ============================================================
# LANGSERVE
# ============================================================

try:

    from langserve import add_routes

    add_routes(
        app,
        graph,
        path="/api",
    )

    logger.info(
        "LangServe mounted at /api"
    )

except Exception as error:

    logger.warning(
        "LangServe could not be mounted: %s",
        error,
    )


# ============================================================
# LOCAL / PRODUCTION ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000",
        )
    )

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )