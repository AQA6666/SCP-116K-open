# SCP-116K Dataset Pipeline

This repository contains the code implementation for the paper: "SCP-116K: A High-Quality Problem-Solution Dataset and a Generalized Pipeline for Automated Extraction in the Higher Education Science Domain"

*(Paper link will be added soon)*

Dataset available at: [https://huggingface.co/datasets/EricLu/SCP-116K](https://huggingface.co/datasets/EricLu/SCP-116K)

## Pipeline Overview

This is a generalized pipeline for automatically extracting high-quality problem-solution pairs from various publicly available documents crawled from the internet. The pipeline consists of the following steps:

1. `fileter_pcb_textbook_and_problem_book_from_lib_meta.py`
   - Filter and identify potential textbooks and problem books from library metadata

2. `transfer_pdf_to_text_with_4o.py`
   - Convert PDF documents to text format with enhanced OCR capabilities

3. `get_book_page_unit_start_index.py`
   - Generate page and unit indices for better content organization

4. `split_book_to_chunk_by_llm_index.py`
   - Split books into manageable chunks using LLM-based indexing

5. `extract_problem_and_solution_from_book_text.py`
   - Extract potential problem-solution pairs from the processed text

6. `filter_problem_and_solution.py`
   - Filter and validate the extracted problem-solution pairs

7. `recall_solutions_for_problems.py`
   - Match problems with their corresponding solutions

8. `judge_problems_and_solutions_match.py`
   - Verify and validate the matched problem-solution pairs

## Usage

For detailed information about each step and how to use the pipeline, please refer to:
- The individual Python files in this repository
- The research paper (link will be added soon)

## Citation

*(Citation information will be added when the paper is published)*

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 