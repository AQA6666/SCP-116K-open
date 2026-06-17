# SCP-116K Dataset Pipeline

This repository contains the code implementation for the paper: "SCP-116K: A High-Quality Problem-Solution Dataset and a Generalized Pipeline for Automated Extraction in the Higher Education Science Domain"

[Paper Link](https://arxiv.org/abs/2501.15587)

Dataset available at: [https://huggingface.co/datasets/EricLu/SCP-116K](https://huggingface.co/datasets/EricLu/SCP-116K)

## Pipeline Overview

This is a generalized pipeline for automatically extracting high-quality problem-solution pairs from various publicly available documents crawled from the internet. The pipeline consists of the following steps:

1. `transfer_pdf_to_text_with_vlm.py`
   - OCR page text from PDF-rendered images using a vision-language model (VLM)

2. `get_book_page_unit_start_index.py`
   - Use an LLM to identify which lines on each page are the start of independent units (e.g., chapters and sections)

3. `split_book_to_chunk_by_llm_index.py`
   - Re-chunk page-level text into roughly equal-length blocks based on the unit indices from step 2 and a token-length threshold

4. `extract_problem_and_solution_from_book_text.py`
   - Extract problems and solutions from the text chunks

5. `recall_solutions_for_problems.py`
   - For each problem within a document, recall K candidate solutions based on problem number and text similarity

6. `judge_problems_and_solutions_match_async.py`
   - Use an LLM to determine which of the recalled solutions in step 5 is the true match for each problem

7. `filter_problem_had_matched_solutions.py`
   - Keep only problems that have a matched solution from step 6

8. `filter_problem_and_solution.py`
   - Filter problem-solution pairs to retain only complete and valid data

## Supporting Files

The pipeline scripts share the following utilities and infrastructure:

- `utils.py`
  - Common file I/O helpers used across the pipeline, including `load_jsonl`, `write_jsonl`, and `find_files` for reading/writing JSONL data and locating files by pattern

- `gpt4_request.py`
  - Wrapper functions for calling OpenAI-compatible APIs, including text-only requests, single/multi-image VLM requests, and retry logic. API credentials are loaded from a `.env` file via `OPENAI_BASE_URL` and `OPENAI_API_KEY`

- `open_vllm_serving.sh`
  - Shell script to launch a vLLM OpenAI-format API server for self-hosted model inference. Configure `CUDA_VISIBLE_DEVICES`, `PORT_NUM`, `MODEL_PATH`, `MODEL_NAME`, and `TP` (tensor parallelism) before running

## Environment Setup

This repository supports two deployment scenarios. Use separate Python environments for each:

### Option A: External OpenAI-compatible API

Use this when pipeline scripts call a remote or third-party API (e.g., OpenAI, Azure OpenAI, or any hosted OpenAI-format endpoint). 

```bash
pip install -r requirements-api.txt
```

Create a `.env` file in the project root for API access (used by `gpt4_request.py` and scripts that call LLMs):

```
OPENAI_BASE_URL=https://your-api-endpoint/v1/
OPENAI_API_KEY=your-api-key
```

Point each script's `--base_url` / `--api_key` arguments (where applicable) to the same endpoint.

### Option B: Local vLLM deployment

Use this when you want to serve open-source models locally on GPU machines. Install dependencies in a **separate** environment on the serving machine:

```bash
pip install -r requirements-vllm.txt
```

Then configure and run `open_vllm_serving.sh`. After the server is up, run the pipeline scripts (Option A environment) with `OPENAI_BASE_URL` set to the local vLLM endpoint, e.g. `http://localhost:8998/v1/`.

| File | Purpose |
|------|---------|
| `requirements-api.txt` | Run pipeline scripts; call external or local OpenAI-format APIs |
| `requirements-vllm.txt` | Serve open-source models locally via vLLM |

**Notes:**
- `requirements-api.txt` includes `torch` and `sentence-transformers` for embedding-based solution recall in step 5.
- vLLM serving and pipeline execution can run on different machines; only the API endpoint needs to be reachable.
- Recommended Python version: **3.10+**

## Usage

For detailed information about each step and how to use the pipeline, please refer to:
- The individual Python files in this repository
- The supporting files above for API access and data I/O
- The research paper

## Citation

```bibtex
@misc{lu2025scp116khighqualityproblemsolutiondataset,
      title={SCP-116K: A High-Quality Problem-Solution Dataset and a Generalized Pipeline for Automated Extraction in the Higher Education Science Domain}, 
      author={Dakuan Lu and Xiaoyu Tan and Rui Xu and Tianchu Yao and Chao Qu and Wei Chu and Yinghui Xu and Yuan Qi},
      year={2025},
      eprint={2501.15587},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.15587}, 
}
```

## License

Dataset is licensed under the [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
Code is licensed under the [MIT License](https://opensource.org/licenses/MIT).

Copyright (c) 2024