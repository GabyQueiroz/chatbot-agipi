# Publicacao do Chatbot AGIPI

Este projeto ficou pronto para ser publicado de duas formas:

- `Streamlit Community Cloud`: mais simples para demonstracao e compartilhamento rapido.
- `Docker`: melhor para servidor proprio, uso institucional e mais controle.

## 1. Publicar no GitHub

No PowerShell, dentro da pasta do projeto:

```powershell
git init
git add .
git commit -m "Preparar chatbot AGIPI para publicacao"
```

Depois crie um repositorio no GitHub e conecte:

```powershell
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git branch -M main
git push -u origin main
```

## 2. Publicar no Streamlit Community Cloud

1. Acesse [https://share.streamlit.io](https://share.streamlit.io)
2. Entre com sua conta GitHub.
3. Clique em `New app`.
4. Selecione o repositorio publicado.
5. Em `Main file path`, use:

```text
app_chatbot_faiss.py
```

6. Clique em `Deploy`.

## 3. Publicar com Docker

No PowerShell:

```powershell
docker build -t chatbot-agipi .
docker run -p 8501:8501 chatbot-agipi
```

Depois acesse:

```text
http://localhost:8501
```

## 4. Observacoes importantes

- O projeto usa a base local `documentos_texto.json`, a planilha `faq_agipi_ageuni_documentos.xlsx`, a planilha `Perguntas.xlsx` e os artefatos de `saida/prepared_corpus` e `saida/faiss_rag_index`.
- Esses arquivos precisam permanecer junto do projeto para o app funcionar corretamente.
- Os caches grandes em `joblib` foram excluidos do pacote de publicacao porque nao sao necessarios para o app publicado.
- Se voce for publicar em ambiente publico, confirme antes se os documentos podem ser expostos.
- Em servidor com menos recursos, pode ser melhor deixar o uso de LLM desativado com a variavel:

```powershell
$env:RAG_FAISS_USE_LLM='0'
```

## 5. Arquivos de deploy preparados

- `Dockerfile`
- `.dockerignore`
- `.gitignore`
- `.streamlit/config.toml`
- `runtime.txt`

## 6. Arquivo principal publicado

O app principal para publicacao e:

- `app_chatbot_faiss.py`
