# =============================================================================
# app_db.py  — with HITL approval UI for purchase_stock
# =============================================================================

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
import streamlit as st
import uuid
import os

from backend import get_all_threads, ingest_rag_document

st.title("Agentic ChatBot")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    st.session_state["message_history"] = []
    st.session_state["chat_titles"][new_id] = "New Chat"
    st.session_state["pending_interrupt"] = None   # clear any pending approval


def load_conversation(thread_id):
    state = workflow.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


def get_title_from_messages(messages):
    for message in messages:
        if isinstance(message, HumanMessage) and message.content.strip():
            title = message.content.strip()
            return title if len(title) <= 40 else title[:37] + "..."
    return "New Chat"


def load_all_threads():
    try:
        seen = set(st.session_state["chat_threads"])
        for checkpoint_tuple in workflow.checkpointer.list(config={}):
            thread_id = (
                checkpoint_tuple.config
                .get("configurable", {})
                .get("thread_id")
            )
            if not thread_id or thread_id in seen:
                continue
            state = workflow.get_state(
                config={"configurable": {"thread_id": thread_id}}
            )
            messages = state.values.get("messages", [])
            if not messages:
                continue
            st.session_state["chat_threads"].append(thread_id)
            seen.add(thread_id)
            if thread_id not in st.session_state["chat_titles"]:
                st.session_state["chat_titles"][thread_id] = (
                    get_title_from_messages(messages)
                )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Load workflow
# ---------------------------------------------------------------------------

@st.cache_resource
def load_workflow():
    from backend import workflow
    return workflow

workflow = load_workflow()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

if "thread_id" not in st.session_state:
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    st.session_state["chat_titles"][new_id] = "New Chat"

if "threads_loaded" not in st.session_state:
    load_all_threads()
    st.session_state["threads_loaded"] = True

# pending_interrupt holds the interrupt prompt string when the graph is paused,
# None when the graph is running normally.
if "pending_interrupt" not in st.session_state:
    st.session_state["pending_interrupt"] = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("My Conversations")

if st.sidebar.button("➕ New Conversation"):
    reset_chat()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Upload PDF to Knowledge Base")

uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Add to Knowledge Base"):
        with st.sidebar.status("Processing PDF...", expanded=False) as status:
            try:
                temp_dir = "uploaded_pdfs"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                ingest_rag_document(temp_path)
                status.update(label="✅ PDF added to knowledge base!", state="complete")
                st.sidebar.success(f"'{uploaded_file.name}' indexed successfully.")
            except Exception as e:
                status.update(label="❌ Failed to process PDF", state="error")
                st.sidebar.error(f"Error: {e}")

for thread_id in st.session_state["chat_threads"][::-1]:
    title = st.session_state["chat_titles"].get(thread_id, "New Chat")
    is_active = thread_id == st.session_state["thread_id"]
    label = f"▶ {title}" if is_active else title

    if st.sidebar.button(label, key=thread_id):
        st.session_state["thread_id"] = thread_id
        st.session_state["pending_interrupt"] = None
        messages = load_conversation(thread_id)
        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue
            temp_messages.append({"role": role, "content": message.content})
        st.session_state["message_history"] = temp_messages
        st.rerun()

# ---------------------------------------------------------------------------
# Chat area: render history
# ---------------------------------------------------------------------------

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# HITL Approval UI
# ---------------------------------------------------------------------------
# When the graph is paused at interrupt(), we show an approval card instead
# of the normal chat input. The user clicks Approve or Reject, which resumes
# the graph with Command(resume="yes") or Command(resume="no").

if st.session_state["pending_interrupt"] is not None:
    thread_id = st.session_state["thread_id"]
    CONFIG = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_trace",
    }

    # Show the approval card
    with st.chat_message("assistant"):
        st.warning(f"⏸ **Approval Required**\n\n{st.session_state['pending_interrupt']}")
        col1, col2 = st.columns(2)
        approve_clicked = col1.button("✅ Approve", key="approve_btn", type="primary")
        reject_clicked  = col2.button("❌ Reject",  key="reject_btn")

    if approve_clicked or reject_clicked:
        decision = "yes" if approve_clicked else "no"

        # Resume the paused graph with the human's decision
        try:
            with st.chat_message("assistant"):
                def resume_stream():
                    for message_chunk, metadata in workflow.stream(
                        Command(resume=decision),
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        if not isinstance(message_chunk, AIMessage):
                            continue
                        content = message_chunk.content
                        if isinstance(content, str):
                            if content:
                                yield content
                        elif isinstance(content, list):
                            for block in content:
                                if not isinstance(block, dict):
                                    continue
                                if block.get("type") == "tool_use":
                                    yield f"\n⚙️ *Calling tool: `{block.get('name', 'unknown')}`...*\n\n"
                                elif block.get("type") == "text" and block.get("text"):
                                    yield block["text"]

                ai_message = st.write_stream(resume_stream())

        except Exception as e:
            ai_message = f"Error: {e}"
            st.error(ai_message)

        # Save response, clear the interrupt flag, rerun
        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
        st.session_state["pending_interrupt"] = None
        st.rerun()

# ---------------------------------------------------------------------------
# Normal chat input (hidden while an interrupt is pending)
# ---------------------------------------------------------------------------

else:
    user_input = st.chat_input("Type here")

    if user_input:
        thread_id = st.session_state["thread_id"]

        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        CONFIG = {
            "configurable": {"thread_id": thread_id},
            "metadata": {"thread_id": thread_id},
            "run_name": "chat_trace",
        }

        try:
            with st.chat_message("assistant"):
                def token_stream():
                    for message_chunk, metadata in workflow.stream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        if not isinstance(message_chunk, AIMessage):
                            continue
                        content = message_chunk.content
                        if isinstance(content, str):
                            if content:
                                yield content
                        elif isinstance(content, list):
                            for block in content:
                                if not isinstance(block, dict):
                                    continue
                                if block.get("type") == "tool_use":
                                    yield f"\n⚙️ *Calling tool: `{block.get('name', 'unknown')}`...*\n\n"
                                elif block.get("type") == "text" and block.get("text"):
                                    yield block["text"]

                ai_message = st.write_stream(token_stream())

            # ── Check for interrupt AFTER the stream ends ──────────────────
            # workflow.stream() stops yielding silently when interrupt() fires
            # inside a tool — it does NOT raise an exception. So we inspect
            # the graph state after streaming to see if it paused mid-run.
            state = workflow.get_state(config=CONFIG)
            if state.tasks:
                interrupts = state.tasks[0].interrupts
                if interrupts:
                    prompt = interrupts[0].value
                    st.session_state["pending_interrupt"] = prompt
                    # Replace the streamed message with a pending notice
                    ai_message = "⏸ Waiting for your approval to proceed..."

        except Exception as e:
            ai_message = f"Error: {e}"
            st.error(ai_message)

        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

        is_new_thread = thread_id not in st.session_state["chat_threads"]
        add_thread(thread_id)

        if is_new_thread:
            title = user_input.strip()
            st.session_state["chat_titles"][thread_id] = (
                title if len(title) <= 40 else title[:37] + "..."
            )

        st.rerun()
