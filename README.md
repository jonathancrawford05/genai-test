# Generative AI Test

## Role Context

The Generative AI role at ZestyAI focuses on building, evaluating, and improving **Retrieval-Augmented Generation (RAG)** pipelines and agentic workflows that work with both structured and unstructured data. It involves implementing solutions, running experiments, developing evaluation frameworks, and iterating to improve performance.

You are **not expected to know everything**. There will be opportunities to learn new tools, techniques, and frameworks as you go. You will also encounter challenges that don’t have predefined solutions, and part of the role is figuring out how to approach these problems.

The purpose of this test is to give you a taste of the types of problems you might see at ZestyAI. The questions here were carefully designed to reflect **real work** in this role. If you enjoy this test, there’s a good chance you’ll enjoy the work itself.

## Overview

The goal of this test is to create a prototype solution that can answer questions about PDF documents. The PDFs may be long and contain complex content such as tables, lists, and text spread across multiple pages.

This test has **two parts**:

* **Part 1:** Build a function to answer questions about PDFs.
* **Part 2:** Build a basic experimentation harness to test multiple solution ideas from Part 1.

This test focuses on prototyping a solution. You are not expected to deliver production-level code. Your solution is meant to be a prototype that can run in a notebook or a Python script.

---

## Part 1: PDF Question Answering

### Objective

Create a function that answers questions based on PDF documents. The function will receive a folder containing the documents and a specific question, and it should answer that question correctly. You may use any agentic design pattern of your choice.

You will be given 3 questions and their expected answers.

### Updated Function Signature

```python
def answer_pdf_question(question: str, pdfs_folder: str) -> str:
    """
    Inputs:
        question: A question about the content of the PDFs.
        pdfs_folder: Path to a folder containing all the PDFs needed to answer the question.
    Output:
        answer: Answer to the question.
    """
    pass
```

Here’s an example of how you would call the function:

```python
# Example 1
question_1 = "List all rating plan rules"
pdfs_folder_1 = "artifacts/2"  
answer_1 = answer_pdf_question(question_1, pdfs_folder_1)
```

### Requirements & Guidance

1. **Focus on accuracy:** Ensure your function produces correct answers for the provided PDFs and questions, while also being designed to generalize beyond these specific cases, not relying on one-off solutions.

2. You may use **any tools, frameworks, or LLM calls**. Multiple steps, agents, or tool integrations are acceptable.

3. It is acceptable to use AI tools (e.g., ChatGPT, Cursor, etc.) during development. If you do, please keep a record of the prompts and steps you used. This helps us understand your reasoning and how you arrived at your solution.

4. Optimizing for context efficiency (token usage) and latency (speed) is encouraged, but accuracy remains the top priority. You should also consider the **scalability** and **cost** of your solution (e.g., handling large numbers of PDFs).

You may create API keys as needed, but please let us know which ones you used so we can reproduce your results.

> **Note:** If you are not able to achieve perfect accuracy for all questions, that is fine as long as your approach is systematic and the evaluation metrics clearly show where the solution performs well and where it struggles.

---

## Part 2: Experimentation Harness

### Objective

Design a small framework to evaluate multiple iterations or approaches from Part 1. Experimentation is a core part of improving generative AI systems.

### Guidelines

1. Implement **at least 2 variations** of your PDF QA function from Part 1.
   You may modify a single parameter, such as a prompt update or a workflow setting, for each variation.
   The goal is not to build a completely new pipeline but to demonstrate thoughtful evaluation.
   Each variation should be encapsulated in its own function.

2. Your experimentation harness should:

   * **Run all variations:** Execute each version of your PDF QA function on all provided example PDFs and questions, and record the results.
   * **Choose evaluation metrics:** Select at least one metric you believe is relevant for measuring solution quality. You may include more if useful. Document and justify your choices.
   * **Compare and iterate:** Provide a summary or table comparing results across iterations. Use the results to identify what works best and suggest further improvements.


> When recording results for each variation, be sure to capture **both successes and failures**. The metrics should clearly indicate where the solution fails so we can assess limitations and potential improvements.

---

## Example Questions

You will find the PDFs and the CSV containing the questions inside the `artifacts` folder.

**Note:** For details about how expected answers were determined, please refer to the README file in the artifacts folder.

### CSV Columns

The questions CSV contains:

* **id** — Unique identifier for each question
* **question** — The question to answer
* **expected_output** — The expected answer
* **PDF Folder** — The folder containing the relevant PDF(s)

---

## Deliverables

1. A Python file or notebook implementing **Part 1** and **Part 2**.
2. Example runs showing answers for all provided PDFs and questions.
3. (Optional) Notes on prompts and experimentation steps if you used AI tools.
4. **Documentation of decisions:** Include comments describing key decisions (e.g., evaluation metric choice, PDF processing approach, agent design). This helps us understand your reasoning.
5. Ensure your code is fully runnable by including clear setup instructions.

---

## Thanks

We recognize that this test requires a significant amount of work. We hope you find it interesting and challenging. Thank you for your time and effort.

This test is a work in progress. If anything is unclear or you have any suggestions, please reach out.

