from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from rag_chatbot_faiss import build_faiss_chatbot_from_env


BASE_PATH = Path(__file__).resolve().parent


@st.cache_data
def load_question_bank(base_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    questions_path = base_path / "Perguntas.xlsx"
    if questions_path.exists():
        df = pd.read_excel(questions_path)
        for column in ("Pergunta", "Categoria"):
            if column not in df.columns:
                df[column] = ""
        frames.append(df[["Pergunta", "Categoria"]].fillna(""))

    faq_path = base_path / "faq_agipi_ageuni_documentos.xlsx"
    if faq_path.exists():
        raw = pd.read_excel(faq_path, header=None)
        header_row = None
        for idx, row in raw.iterrows():
            first = str(row.iloc[0]).strip() if len(row) > 0 else ""
            second = str(row.iloc[1]).strip() if len(row) > 1 else ""
            if first == "Pergunta" and second == "Categoria":
                header_row = idx
                break
        if header_row is not None:
            headers = [str(value).strip() for value in raw.iloc[header_row].tolist()]
            faq_df = raw.iloc[header_row + 1 :].copy()
            faq_df.columns = headers
            for column in ("Pergunta", "Categoria"):
                if column not in faq_df.columns:
                    faq_df[column] = ""
            frames.append(faq_df[["Pergunta", "Categoria"]].fillna(""))

    if not frames:
        return pd.DataFrame(columns=["Pergunta", "Categoria"])

    combined = pd.concat(frames, ignore_index=True)
    combined["Pergunta"] = combined["Pergunta"].astype(str).str.strip()
    combined["Categoria"] = combined["Categoria"].astype(str).str.strip()
    combined = combined[combined["Pergunta"] != ""].drop_duplicates(subset=["Pergunta", "Categoria"]).reset_index(drop=True)
    return combined


st.set_page_config(page_title="Chatbot AGIPI", page_icon="💬", layout="wide")

st.markdown("<h1 style='font-size: 3rem; font-weight: 800; margin-bottom: 0.25rem;'>Chatbot AGIPI</h1>", unsafe_allow_html=True)

question_bank = load_question_bank(BASE_PATH)

if "faiss_chatbot" not in st.session_state:
    with st.spinner("Carregando base, embeddings e indice FAISS..."):
        st.session_state.faiss_chatbot = build_faiss_chatbot_from_env(BASE_PATH)

if "faiss_messages" not in st.session_state:
    st.session_state.faiss_messages = [
        {
            "role": "assistant",
            "content": (
                "Voce pode me perguntar sobre AGIPI, UEPG, AGEUNI, propriedade intelectual, incubacao, "
                "transferencia de tecnologia, inovacao ou normas do INPI. Se quiser, pode perguntar de um jeito direto "
                "ou mais conversado."
            ),
        }
    ]

with st.sidebar:
    if not question_bank.empty:
        st.subheader("Perguntas")
        categories = sorted(category for category in question_bank["Categoria"].unique().tolist() if category)
        selected_category = st.selectbox("Categoria", options=["Todas"] + categories, index=0)
        filtered = question_bank if selected_category == "Todas" else question_bank[question_bank["Categoria"] == selected_category]
        for idx, item in enumerate(filtered["Pergunta"].tolist()):
            if st.button(item, key=f"faiss_question_{selected_category}_{idx}", use_container_width=True):
                st.session_state["faiss_prefilled_prompt"] = item

    retrieve_k = st.slider("Trechos recuperados", min_value=4, max_value=20, value=8, step=1)
    st.caption("Quanto maior o valor, mais contexto o FAISS traz para a resposta.")

for message in st.session_state.faiss_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("follow_up_suggestions"):
            st.caption("Voce tambem pode perguntar: " + " | ".join(message["follow_up_suggestions"][:3]))

prompt = st.chat_input("Digite sua pergunta")

if st.session_state.get("faiss_prefilled_prompt"):
    prompt = st.session_state.pop("faiss_prefilled_prompt")

if prompt:
    st.session_state.faiss_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = st.session_state.faiss_messages[:-1]
    with st.chat_message("assistant"):
        with st.spinner("Consultando o indice FAISS e gerando resposta..."):
            result = st.session_state.faiss_chatbot.answer(prompt, history=history, retrieve_k=retrieve_k)
        st.markdown(result["answer"])
        if result.get("follow_up_suggestions"):
            st.caption("Voce tambem pode perguntar: " + " | ".join(result["follow_up_suggestions"][:3]))

    st.session_state.faiss_messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "suggested_category": result.get("suggested_category", ""),
            "similar_questions": result.get("similar_questions", []),
            "follow_up_suggestions": result.get("follow_up_suggestions", []),
        }
    )
