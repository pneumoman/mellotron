#!/usr/bin/env bash

python -m pip install --user ipywidgets
jupyter nbextension enable --py widgetsnbextension
jupyter notebook --ip 0.0.0.0