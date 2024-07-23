install:
	poetry install
	python -m spacy download xx_ent_wiki_sm
	transformers-cli download --cache-dir "./models" mixedbread-ai/mxbai-embed-large-v1

dev:
	poetry run uvicorn app:app --host 127.0.0.1 --port 8000 --reload

run:
	poetry run uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info --workers 4

shell:
	@echo 'Starting poetry shell. Press Ctrl-d to exit from the shell'
	poetry shell

server_restart: 
	systemctl daemon-reload
	systemctl start fastapi

worker: 
	dramatiq worker --processes 4 --threads 4