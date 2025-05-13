name: Python Otomatik Test

on: [push, pull_request]  # Her push ve PR'da çalışır

jobs:
  test:
    runs-on: ubuntu-latest  # Sanal makine

    steps:
    - name: Repo dosyalarını çek
      uses: actions/checkout@v3

    - name: Python ortamını kur
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Bağımlılıkları yükle
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

    - name: Testleri çalıştır
      run: |
        python -m unittest discover
