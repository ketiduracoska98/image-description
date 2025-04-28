# IBD-PROJECT

How to run:
1. Setup for ofa-model - open a terminal and execute following commands:
cd ofa-app
python -m venv ofa-venv
.\ofa-venv\Scripts\activate
git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
pip install OFA/transformers/
git clone https://huggingface.co/OFA-Sys/OFA-large-caption
pip install flask
pip install pillow
pip install torchvision
python -m flask run --port=5001  (alternative: python .\app.py)

2. Run app: on a new terminal (in IBD-PROJECT folder), run:
python -m flask run    (alternative: python .\app.py)


