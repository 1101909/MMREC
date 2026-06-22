# Kaggle runners for MMRec

## Clone from GitHub and run

Use this cell in a Kaggle notebook:

```python
!rm -rf /kaggle/working/MMREC
!git clone https://github.com/1101909/MMREC.git /kaggle/working/MMREC
%cd /kaggle/working/MMREC

!python kaggle_run_all_selected_models.py \
  --data-path /kaggle/input/datasets/toanktx/mmrec-cold \
  --datasets baby sports clothing elec \
  --models VBPR LightGCN MMGCN GRCN LATTICE FREEDOM BM3 SLMRec LGMRec MGCN SMORE \
  --epochs 5
```

For a short smoke run:

```python
!python kaggle_run_all_selected_models.py \
  --data-path /kaggle/input/datasets/toanktx/mmrec-cold \
  --dataset baby \
  --models BM3 \
  --epochs 1
```

The runner reads source data from `/kaggle/input/datasets/toanktx/mmrec-cold`,
prepares writable fixed data under `/kaggle/working/mmrec-cold-fixed`, and
writes logs plus summary CSV files under `src/kaggle_run_results`.

## One-cell model runners

Each `.py` file in this folder is standalone. Copy the whole content of one file
into a Kaggle notebook cell and run it.

Default behavior per file:
- runs exactly one model named by the file
- runs datasets: `baby`, `sports`, `clothing`, `elec`
- reads source data from `/kaggle/input/datasets/toanktx/mmrec-cold`
- prepares fixed data under `/kaggle/working/mmrec-cold-fixed`
- clones and patches MMRec under `/kaggle/working/MMRec`
- writes logs and summary CSV under `/kaggle/working/MMRec/src/kaggle_run_results`
- prints parsed metrics after each dataset run and prints a final results table

To run one dataset only, edit or add before `main()` is called:

```python
import sys
sys.argv = ["runner", "--dataset", "baby", "--epochs", "5"]
```
