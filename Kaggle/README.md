# Kaggle one-cell runners for MMRec

Each `.py` file in this folder is standalone. Copy the whole content of one file into a Kaggle notebook cell and run it.

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
