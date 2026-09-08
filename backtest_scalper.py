name: "Correr Backtest Walk-Forward"

on:
  workflow_dispatch:

permissions:
  contents: write

jobs:
  run_evaluations:
    name: Validacion Cruzada Out-of-Sample
    runs-on: ubuntu-latest
    steps:
    - name: Clonar Repositorio
      uses: actions/checkout@v4

    - name: Configurar Entorno Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'

    - name: Instalar Librerias Base
      run: pip install pandas numpy requests

    - name: Sanear Estructuras CSV Ocultas
      run: python limpiar_headers.py

    - name: Procesar Evaluacion Completa
      env:
        TELEGRAM_TOKEN: ${{ secrets.TELEGRAM_TOKEN }}
        TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
      run: |
        python backtest_diario.py
        python backtest_scalper.py

    - name: Sincronizar Cambios de Limpieza en GitHub
      run: |
        git config --global user.name 'github-actions[bot]'
        git config --global user.email 'github-actions[bot]@://github.com'
        git add soxl_predictions.csv scalper_predictions.csv || true
        git commit -m "Auto-cleanup: Reset de Cabeceras CSV" || true
        git push || true
