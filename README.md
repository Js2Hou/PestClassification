# Image recognition of leaf vegetable pests and diseases

This is the baseline project for Image recognition of leaf vegetable pests and diseases.

## Fast to start

run train.py to train finetune resnet50, after that run test.py to generate predicted results saved in '/results/test.csv'.

## Directory structure

```python
│PestClassification/
├──data/
├──results/
│  ├── test.csv
├──models/
│  ├── __init__.py
│  ├── _resnet50.py
├──tb_logs/
├──checkpoints/
├──load_data.py
├──train.py
├──test.py
├──utils.py
├──README.md

```

## Todo

- try **volo**
- retrain resnet50 and test again
