.PHONY: help setup install test clean format lint check-env

# Cores para formatação
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

# Variáveis
VENV_NAME?=venv
PYTHON=${VENV_NAME}/Scripts/python
PIP=${VENV_NAME}/Scripts/pip

## Exibe esta ajuda
help:
	@echo '\n${YELLOW}Uso:${RESET}'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo '${YELLOW}Targets:${RESET}'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  ${YELLOW}%-20s${GREEN}%s${RESET}\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

## Cria ambiente virtual e instala dependências
setup: ## Cria o ambiente virtual e instala as dependências
	@echo "${YELLOW}Criando ambiente virtual...${RESET}"
	python -m venv $(VENV_NAME)
	@echo "${YELLOW}Atualizando pip...${RESET}"
	${PYTHON} -m pip install --upgrade pip
	@echo "${YELLOW}Instalando dependências...${RESET}"
	${PIP} install -r requirements.txt
	@echo "${GREEN}✓ Ambiente configurado com sucesso!${RESET}"

## Instala dependências
install: ## Instala as dependências do projeto
	@echo "${YELLOW}Instalando dependências...${RESET}"
	${PIP} install -r requirements.txt

## Ativa o ambiente virtual
activate: ## Ativa o ambiente virtual
	@echo "${YELLOW}Para ativar o ambiente virtual, execute:${RESET}"
	@echo "${GREEN}source ${VENV_NAME}/Scripts/activate${RESET} (Linux/Mac)"
	@echo "${GREEN}${VENV_NAME}\\Scripts\\activate${RESET} (Windows)"

## Formata o código
format: ## Formata o código usando black e isort
	@echo "${YELLOW}Formatando código...${RESET}"
	${PYTHON} -m black .
	${PYTHON} -m isort .

## Verifica a qualidade do código
lint: ## Verifica a qualidade do código com flake8
	@echo "${YELLOW}Verificando qualidade do código...${RESET}"
	${PYTHON} -m flake8 .

## Executa testes
# test: ## Executa os testes do projeto
# 	@echo "${YELLOW}Executando testes...${RESET}"
# 	${PYTHON} -m pytest tests/

## Limpa arquivos temporários
clean: ## Remove arquivos temporários e cache
	@echo "${YELLOW}Limpando arquivos temporários...${RESET}"
	rm -rf `find . -type d -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -f `find . -type f -name '*~'`
	rm -f `find . -type f -name '.*~'`
	rm -rf .pytest_cache
	rm -rf .mypy_cache

## Remove o ambiente virtual
distclean: clean ## Remove o ambiente virtual e limpa tudo
	@echo "${YELLOW}Removendo ambiente virtual...${RESET}"
	rm -rf $(VENV_NAME)

## Verifica se o ambiente virtual está ativo
check-env:
	@if [ -z "${VIRTUAL_ENV}" ]; then \
		echo "${YELLOW}Atenção: Ambiente virtual não ativado.${RESET}"; \
		echo "Execute '${GREEN}make setup${RESET}' para configurar e ativar o ambiente virtual."; \
		exit 1; \
	fi

## Inicia o Jupyter Notebook
notebook: check-env ## Inicia o Jupyter Notebook
	@echo "${YELLOW}Iniciando Jupyter Notebook...${RESET}"
	jupyter notebook

## Exporta as dependências
freeze: ## Gera o arquivo requirements.txt
	@echo "${YELLOW}Atualizando requirements.txt...${RESET}"
	${PIP} freeze > requirements.txt
	@echo "${GREEN}✓ requirements.txt atualizado com sucesso!${RESET}"
