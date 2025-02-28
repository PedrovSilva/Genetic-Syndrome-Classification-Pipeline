# Genetic Syndrome Classification Pipeline

This repository contains a pipeline for the classification of genetic syndromes based on image embeddings.

## Project Structure

```
├── data/
│   ├── input/                # Input files
│   │   ├── mini_gm_public_v0.1.p  # Input data
│   ├── output/               # Pipeline results
│   │   ├── tables/           # Generated tables
│   │   ├── plots/            # Generated plots
├── main.py                    # Main script
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Requirements

Ensure that Python is installed in your environment.

### Installing Dependencies

Run the following command to install the necessary dependencies:

```sh
pip install -r requirements.txt
```

## Execution

To run the pipeline, execute:

```sh
python main.py
```

## Output

After execution, the results will be available in the following directories:
- **Tables**: `data/output/tables/`
- **Plots**: `data/output/plots/`



