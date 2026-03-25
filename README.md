# Chatbot AGIPI

Aplicacao em Streamlit para responder perguntas sobre AGIPI, UEPG, AGEUNI, propriedade intelectual, incubacao, transferencia de tecnologia e normas relacionadas.

## App principal

O app pronto para publicacao e:

- `app_chatbot_faiss.py`

O motor principal e:

- `rag_chatbot_faiss.py`

## Como rodar localmente

```powershell
pip install -r requirements.txt
streamlit run app_chatbot_faiss.py
```

## Base utilizada

O projeto usa:

- `documentos_texto.json`
- `faq_agipi_ageuni_documentos.xlsx`
- `Perguntas.xlsx`
- `saida/prepared_corpus`
- `saida/faiss_rag_index`

## Publicacao

O projeto foi preparado para publicacao com:

- GitHub
- Streamlit Community Cloud
- Docker

As instrucoes completas estao em:

- `PUBLICACAO.md`

## Arquivos de deploy incluidos

- `.gitignore`
- `.dockerignore`
- `.streamlit/config.toml`
- `Dockerfile`
- `runtime.txt`

## Observacao

Se o ambiente de publicacao tiver poucos recursos, voce pode rodar sem LLM local:

```powershell
$env:RAG_FAISS_USE_LLM='0'
streamlit run app_chatbot_faiss.py
```
