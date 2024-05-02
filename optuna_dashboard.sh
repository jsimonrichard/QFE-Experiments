#!/bin/sh
export $(grep -v '^#' .env | xargs)
optuna-dashboard $OPTUNA_DB