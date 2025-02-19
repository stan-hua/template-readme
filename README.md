
<!-- OPTIONAL: Project Logo
<p align="center">
  <img src="[OPT FILL: Path/link to logo image]" alt="Logo" style="width: 15%; display: block; margin: auto;">
</p>
-->

<h1 align="center"> [MUST FILL: Project Name] </h1>


<!-- OPTIONAL: Badges with Hyperlinks
<p align="center">
  <a href="[OPT FILL: Path/link to paper]"><img src="https://img.shields.io/badge/arXiv-2405.01535-b31b1b.svg" alt="arXiv"></a>
  <a href="[OPT FILL: Path/link to HuggingFace]"><img src="https://img.shields.io/badge/Hugging%20Face-Organization-ff9d00" alt="Hugging Face Organization"></a>
  <a href="[OPT FILL: Path/link to LICENSE]"><img src="https://img.shields.io/license-MIT-blue/license-MIT-blue.svg" alt="License"></a>
  <a href="[OPT FILL: Path/link to PyPI project]"><img src="https://img.shields.io/pypi/v/[OPT FILL: Name of PyPI package].svg" alt="PyPI version"></a>
</p>
-->

<p align="center">
  ⚡ A repository for [MUST FILL: 1-line description] 🚀 ⚡ <br>
</p>

**CCM Members**: [MUST FILL: Name of CCM Members with Emails]
1. [Andrei Turinsky](mailto:andrei.turinsky@sickkids.ca)

**Collaborators**: [MUST FILL: Name of PI(s) & External Collaborators]
1. [Anon PI](mailto:andrei.turinsky@sickkids.ca)

## 🔧 Installation

**(Automatic) Installation with pip:**

```shell
# Option 1. Available on PyPI
pip install [OPT FILL: PyPI package name]

# Option 2. Local Pip Install
git clone https://github.com/[OPT FILL: Path to Repo]
cd [OPT FILL: Repository Name]
pip install -e .
```

**(Manual) Installation**
```shell
# Get repository
git clone https://github.com/[OPT FILL: Path to Repo]
cd [OPT FILL: Repository Name]

# Install dependencies
# Option 1. Pip
pip install -r requirements.txt
# Option 2. Conda
conda env create -f environment.yaml
```

## ⏩ How to Run

### Batch Grading

***Note***: If you have multiple responses to grade, don't use `single_absolute_grade` / `single_relative_grade` - instead, use `absolute_grade` and `relative_grade`! It will give you more than 10x speedup.

```python
# batch absolute grade
instructions = [...]  # List of instructions
responses = [...]  # List of responses
reference_answers = [...]  # List of reference answers
rubric = "..."  # Rubric string

feedbacks, scores = judge.absolute_grade(
    instructions=instructions,
    responses=responses,
    rubric=rubric,
    reference_answers=reference_answers
)

# batch relative grade
instructions = [...]  # List of instructions
responses_from_a = [...]  # List of responses
responses_from_b = [...]
reference_answers = [...]  # List of reference answers
rubric = "..."  # Rubric string

feedbacks, scores = judge.relative_grade(
    instructions=instructions,
    responses_A=responses_from_a,
    responses_B=responses_from_b,
    rubric=rubric,
    reference_answers=reference_answers
)
```

## 🤔 What is Prometheus-Eval?

**Prometheus-Eval**🔥 is a repository that provides a collection of tools for training, evaluating, and using language models specialized in evaluating other language models. The repository includes the following components:

1. The `prometheus-eval` Python package, which provides a simple interface for evaluating instruction-response pairs using Prometheus.
2. Collection of evaluation datasets for training and evaluating Prometheus models.
3. Scripts for training Prometheus models or fine-tuning on custom datasets.

### Prometheus 

**Prometheus**🔥 is a family of open-source language models specialized in evaluating other language models. By effectively simulating human judgments and proprietary LM-based evaluations, we aim to resolve the following issues:

* *Fairness*: Not relying on closed-source models for evaluations!

* *Controllability*: You don’t have to worry about GPT version updates or sending your private data to OpenAI by constructing internal evaluation pipelines

* *Affordability*: If you already have GPUs, it is free to use!

<p align="center">
<img align="center" alt="finegrained-eval" src="assets/finegrained_eval.png" width="550"/>
</p>


## 🚀 What's special about Prometheus?

Compared to the Prometheus 1 models, the Prometheus 2 models support both **direct assessment** (absolute grading) and **pairwise ranking** (relative grading). 

You could switch modes by providing a different input prompt format and system prompt. Within the prompt, you should fill in the instruction, response(s), and score rubrics with your own data. Optionally, you could also add a reference answer which leads to better performance!


<p align="center">
<img align="center" alt="formats" src="assets/formats.png" width="700"/>
</p>

 
## 🏃 Running Prometheus-Eval

### Using the package `prometheus-eval`

The `prometheus-eval` package provides a simple interface for evaluating instruction-response pairs using Prometheus. The package includes the following methods:

- `absolute_grade`: Evaluates a single response based on a given instruction, reference answer, and score rubric. Outputs a score between 1 and 5.
- `relative_grade`: Evaluates two responses based on a given instruction and score rubric. Outputs 'A' or 'B' based on the better response.


### Using the weights from Huggingface Hub 🤗

If you prefer directly working with the weights uploaded in Huggingface Hub, you can directly download the model weights! 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("prometheus-eval/prometheus-7b-v2.0")
tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-7b-v2.0")

ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

ABSOLUTE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{rubric}

###Feedback: """

user_content = ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT.format(...) # Fill the prompt with your data

messages = [
    {"role": "user", "content": user_content},
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])


```

## 📚 Learn more

| Section | Description |
|-|-|
| [BiGGen-Bench Evaluation](BiGGen-Bench/README.md) | Instructions to evaluate your LM in BiGGen-Bench. You could also refer to the implementation for your own evaluation benchmark. |
| [Training Prometheus](train/README.md) | Instructions to replicate Prometheus 2 models. Based on the [alignment-handbook](https://github.com/huggingface/alignment-handbook) repository. |
| [Using Prometheus as a data quality filter](https://huggingface.co/blog/burtenshaw/distilabel-prometheus-2) | Cookbook for using Prometheus 2 as a quality filter in synthetic data generation. Huge thanks to the distilabel team! 🙌 |
| [Using Prometheus as an evaluator in RAG](https://docs.llamaindex.ai/en/latest/examples/cookbooks/prometheus2_cookbook/) | Cookbook for using Prometheus 2 RAG applications. Huge thanks to the LlamaIndex team! 🙌 | 


## 👏 Acknowledgements

The underlying codebase for training originates from Huggingface's [Alignment Handbook](https://github.com/huggingface/alignment-handbook) and [Super Mario Merging](https://github.com/martyn/safetensors-merge-supermario) repository. Also, for inference, it heavily utilizes the [litellm](https://github.com/BerriAI/litellm), [vllm](https://github.com/vllm-project/vllm) and the [transformer](https://github.com/huggingface/transformers) library. Huge thanks to all the contributors for these awesome repositories!! 🙌


## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=prometheus-eval/prometheus-eval&type=Date)](https://star-history.com/#prometheus-eval/prometheus-eval&Date)


## Citation

If you find our work useful, please consider citing our paper!

```bibtex
@misc{kim2024prometheus,
      title={Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models}, 
      author={Seungone Kim and Juyoung Suk and Shayne Longpre and Bill Yuchen Lin and Jamin Shin and Sean Welleck and Graham Neubig and Moontae Lee and Kyungjae Lee and Minjoon Seo},
      year={2024},
      eprint={2405.01535},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
```bibtex
@article{kim2023prometheus,
  title={Prometheus: Inducing Fine-grained Evaluation Capability in Language Models},
  author={Kim, Seungone and Shin, Jamin and Cho, Yejin and Jang, Joel and Longpre, Shayne and Lee, Hwaran and Yun, Sangdoo and Shin, Seongjin and Kim, Sungdong and Thorne, James and others},
  journal={arXiv preprint arXiv:2310.08491},
  year={2023}
}
```
```bibtex
@misc{lee2024prometheusvision,
      title={Prometheus-Vision: Vision-Language Model as a Judge for Fine-Grained Evaluation}, 
      author={Seongyun Lee and Seungone Kim and Sue Hyun Park and Geewook Kim and Minjoon Seo},
      year={2024},
      eprint={2401.06591},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
```bibtex
@misc{kim2024biggen,
      title={The BiGGen Bench: A Principled Benchmark for Fine-grained Evaluation of Language Models with Language Models}, 
      author={Seungone Kim and Juyoung Suk and Ji Yong Cho and Shayne Longpre and Chaeeun Kim and Dongkeun Yoon and Guijin Son and Yejin Cho and Sheikh Shafayat and Jinheon Baek and Sue Hyun Park and Hyeonbin Hwang and Jinkyung Jo and Hyowon Cho and Haebin Shin and Seongyun Lee and Hanseok Oh and Noah Lee and Namgyu Ho and Se June Joo and Miyoung Ko and Yoonjoo Lee and Hyungjoo Chae and Jamin Shin and Joel Jang and Seonghyeon Ye and Bill Yuchen Lin and Sean Welleck and Graham Neubig and Moontae Lee and Kyungjae Lee and Minjoon Seo},
      year={2024},
      eprint={2406.05761},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
