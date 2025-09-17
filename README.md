# 📘 ProSeCCo: Propositional Self-Contradiction Corpus

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

## Overview
**ProSeCCo** is a dataset and codebase for **self-contradiction detection in dialogue**.  
It is the first corpus to combine:
- **Naturally occurring dialogue** (debates, online forums, panel discussions)  
- **Propositional reconstructions** (context-resolved statements derived from *Inference Anchoring Theory*)  

The repository provides:
- The **dataset** (JSONL/TSV format)  
- **Annotation guidelines**  
- **Baseline experiments** and training scripts  
- **Error analysis** and **comparative corpus analysis** tools  
- Inter-annotator agreement (IAA) materials  

---

## Repository Structure

```text
ProSeCCo/
│
├── ErrorAnalysis              # scripts for model error analysis
├── IAA                        # inter-annotator agreement materials and scripts
├── comparativeCorpusAnalysis  # scripts for comparing ProSeCCo vs. DECODE
├── data                       # scripts and files for data processing; ProSeCCo_final contains the corpus in JSON and .csv
├── annotation_guidelines.pdf  # PDF of annotation rules + examples
└── README.md                  # this file
```

---

## Dataset Contents

Each record contains:
- `id`: unique identifier  
- `speaker_id`: anonymized speaker code  
- `locution_1`, `locution_2`: original utterances  
- `proposition_1`, `proposition_2`: context-resolved propositions  
- `label`: `self-contradiction` / `no-contradiction`  
- `source`: original corpus (US2016, QT30, QT50)  
- `nodeset_id`: identifier of the nodeset in the source corpus  

### Example (JSON)
```json
{
"id": "QT30_094",
  "speaker_id": "speaker_063",
  "locution_1": "He was shaking hands with the staff.",
  "locution_2": "He was not.",
  "proposition_1": "Boris Johnson was shaking hands with the staff.",
  "proposition_2": "Boris Johnson was not shaking hands with coronavirus patients in a hospital.",
  "label": "no self-contradiction",
  "source": "QT30",
  "nodeset_id": "18507"
}
```
## Statistics
* 1,327 pairs (685 contradictions / 642 non-contradictions)
* Language: English (UK/US varieties)
* Tokens:
  * Locutions: 26,172
  * Propositions: 34,701
* Average length:
  * Locutions: ~10 tokens
  * Propositions: ~13 tokens
* Inter-Annotator Agreement: κ = 0.63

# License
This dataset and code are released under the CC-BY-SA 4.0 license.
You are free to share and adapt, provided you attribute the creators and share derivatives under the same license.

# Citation
If you use this dataset or code, please cite:

```bibtex
@inproceedings{beyer2026prosecco,
  title     = {ProSeCCo: A Dialogue-Sourced, Propositionalised Corpus for Self-Contradiction Detection},
  author    = {...},
  booktitle = {...},
  year      = {...}
}
```

# Contact
For questions or feedback:
📧 [info@arg.tech] [2579207@dundee.ac.uk]
