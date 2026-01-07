# ds-analys-doodle


## Installing UV and building the project


```
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv

#not necessary if uv.lock is present int he folder

```

## General dependencies

```
uv add streamlit numpy pandas matplotlib
```

## Running

```
cd src
streamlit run main.py --client.toolbarMode=minimal

```