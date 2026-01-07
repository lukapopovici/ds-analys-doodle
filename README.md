# ds-analys-doodle

## Installing UV and Building the Project

Install UV package manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment:
```bash
uv venv
```

Note: Installing dependencies is not necessary if `uv.lock` is present in the folder.

## General Dependencies

Install required packages:
```bash
uv add streamlit numpy pandas matplotlib
```

## Running the Application

Navigate to the source directory and run Streamlit:
```bash
cd src
streamlit run main.py --client.toolbarMode=minimal
```