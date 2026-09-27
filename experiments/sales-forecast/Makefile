.PHONY: help setup install test clean format lint check-env

# Colors for formatting
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

# Variables
VENV_NAME?=venv
PYTHON=${VENV_NAME}/Scripts/python
PIP=${VENV_NAME}/Scripts/pip

## Show this help
help:
	@echo '\n${YELLOW}Uso:${RESET}'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo '${YELLOW}Targets:${RESET}'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  ${YELLOW}%-20s${GREEN}%s${RESET}\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

## Create virtual environment and install dependencies
setup: ## Create the virtual environment and install the dependencies
	@echo "${YELLOW}Criando ambiente virtual...${RESET}"
	python -m venv $(VENV_NAME)
	@echo "${YELLOW}Atualizando pip...${RESET}"
	${PYTHON} -m pip install --upgrade pip
	@echo "${YELLOW}Instalando dependências...${RESET}"
	${PIP} install -r requirements.txt
	@echo "${GREEN}✓ Ambiente configurado com sucesso!${RESET}"

## Install dependencies
install: ## Install the project dependencies
	@echo "${YELLOW}Instalando dependências...${RESET}"
	${PIP} install -r requirements.txt

## Activate the virtual environment
activate: ## Activate the virtual environment
	@echo "${YELLOW}Para ativar o ambiente virtual, execute:${RESET}"
	@echo "${GREEN}source ${VENV_NAME}/Scripts/activate${RESET} (Linux/Mac)"
	@echo "${GREEN}${VENV_NAME}\\Scripts\\activate${RESET} (Windows)"

## Format the code
format: ## Format the code using black and isort
	@echo "${YELLOW}Formatando código...${RESET}"
	${PYTHON} -m black .
	${PYTHON} -m isort .

## Check code quality
lint: ## Check code quality with flake8
	@echo "${YELLOW}Verificando qualidade do código...${RESET}"
	${PYTHON} -m flake8 .

## Run tests
# test: ## Run the project tests
# 	@echo "${YELLOW}Executando testes...${RESET}"
# 	${PYTHON} -m pytest tests/

## Clean temporary files
clean: ## Remove temporary files and cache
	@echo "${YELLOW}Limpando arquivos temporários...${RESET}"
	rm -rf `find . -type d -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -f `find . -type f -name '*~'`
	rm -f `find . -type f -name '.*~'`
	rm -rf .pytest_cache
	rm -rf .mypy_cache

## Remove the virtual environment
distclean: clean ## Remove the virtual environment and clean everything
	@echo "${YELLOW}Removendo ambiente virtual...${RESET}"
	rm -rf $(VENV_NAME)

## Check whether the virtual environment is active
check-env:
	@if [ -z "${VIRTUAL_ENV}" ]; then \
		echo "${YELLOW}Atenção: Ambiente virtual não ativado.${RESET}"; \
		echo "Execute '${GREEN}make setup${RESET}' para configurar e ativar o ambiente virtual."; \
		exit 1; \
	fi

## Start the Jupyter Notebook
notebook: check-env ## Start the Jupyter Notebook
	@echo "${YELLOW}Iniciando Jupyter Notebook...${RESET}"
	jupyter notebook

## Export the dependencies
freeze: ## Generate the requirements.txt file
	@echo "${YELLOW}Atualizando requirements.txt...${RESET}"
	${PIP} freeze > requirements.txt
	@echo "${GREEN}✓ requirements.txt atualizado com sucesso!${RESET}"
