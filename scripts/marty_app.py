import streamlit as st
from chat import ask_marty  # this should wrap your full_prompt + GPT call

st.set_page_config(page_title="MartyGPT", page_icon="💬")
st.title("💬 MartyGPT")
st.caption("Chat with your actual boyfriend, simulated by GPT-4 + your text history.")

# Initialize chat history
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# Use a form to handle submission and input reset
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Type your message to Marty...", key="user_input")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    # Add Sophie’s message
    st.session_state.chat_log.append({"sender": "Sophie", "text": user_input})

    # Get Marty’s response
    marty_reply = ask_marty(user_input, st.session_state.chat_log)
    st.session_state.chat_log.append({"sender": "Marty", "text": marty_reply})

# Display chat history
for msg in st.session_state.chat_log:
    if msg["sender"] == "Sophie":
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"<span style='color:#1c6d9f'><b>Marty:</b> {msg['text']}</span>", unsafe_allow_html=True)
