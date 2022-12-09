#!/bin/sh
gunicorn --chdir app capp:app -w 2 -b 0.0.0.0:8050