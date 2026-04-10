## Folder hierarchy

prediction_model

```
├── MANIFEST.in
├── prediction_model
│   ├── config
│   │   ├── config.py
│   │   └── __init__.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── test.csv
│   │   └── train.csv
│   ├── __init__.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── processing
│   │   ├── data_handling.py
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── trained_models
│   │   ├── classification.pkl
│   │   └── __init__.py
│   ├── training_pipeline.py
│   └── VERSION
├── README.md
├── requirements.txt
├── setup.py
└── tests
    ├── pytest.ini
    └── test_prediction.py
```


## Virtual environment
1. Install virtualenv - `python3 -m pip install virtualenv`
2. Check version - `virtualenv --version`
3. Create virtual environment - `virtualenv env_name` (ml_package)
4. Activate virtual environment 
    - For linux/Mac: `source ml_package/bin/activate` (ml_package)
    - For Windows: `source ml_package\Script\activate` (ml_package)
5. Test virtual environment by installing the libraries using *requirements.txt*
    - `pip install -r requirements.txt`
6. Run the training pipeline to test the environment
7. Type `deactivate` the virtual environment
