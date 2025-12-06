"""
Exam Buddy Module
Provides specialized study coaching for Indian competitive exams (JEE, NEET, etc.)
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from api_key_rotator import get_api_key
import logging

logger = logging.getLogger("zenark.exam_buddy")

# System prompt for exam buddy
EXAM_BUDDY_SYSTEM_PROMPT = """You are a friendly and knowledgeable study coach specialized in helping Indian teenage students prepare for competitive exams like JEE Main, NEET, IIT, NIT, etc.

Your expertise includes:
- Effective study techniques and time management
- Memory enhancement tricks for formulas, equations, and periodic tables
- Subject-specific strategies for Chemistry, Mathematics, Physics, and Biology
- Exam preparation psychology and stress management
- Indian education system specific advice

Always be:
- Encouraging and supportive
- Practical with actionable advice
- Culturally aware of Indian student challenges
- Focused on proven study methods

Current user context: {context}

Important: Tailor your advice to competitive exam preparation and provide specific, implementable tips."""

# In-memory session storage (for production, use MongoDB)
_session_store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Retrieve or create chat history for a session.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        ChatMessageHistory object for the session
    """
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def create_exam_buddy_chain():
    """
    Create the exam buddy conversational chain with memory.
    
    Returns:
        RunnableWithMessageHistory chain
    """
    # Initialize LLM with API key rotation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=get_api_key())
    
    # Create prompt template with history
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXAM_BUDDY_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{question}")
    ])
    
    # Create chain
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    # Wrap with message history
    conversational_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )
    
    return conversational_chain


# Global chain instance
_exam_buddy_chain = None


def get_exam_buddy_chain():
    """Get or create the global exam buddy chain instance."""
    global _exam_buddy_chain
    if _exam_buddy_chain is None:
        _exam_buddy_chain = create_exam_buddy_chain()
    return _exam_buddy_chain


async def get_exam_buddy_response(
    question: str,
    session_id: str = "default",
    context: str = ""
) -> str:
    """
    Get a response from the exam buddy.
    
    Args:
        question: User's question about exam preparation
        session_id: Session identifier for conversation history
        context: Additional context about the user
        
    Returns:
        Exam buddy's response as a string
    """
    try:
        chain = get_exam_buddy_chain()
        
        response = await chain.ainvoke(
            {
                "question": question,
                "context": context
            },
            config={
                "configurable": {"session_id": session_id}
            }
        )
        
        logger.info(f"Exam buddy response generated for session {session_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating exam buddy response: {e}")
        return (
            "I'm having trouble right now, but I'm here to help! "
            "Could you rephrase your question about exam preparation?"
        )


def clear_session_history(session_id: str):
    """
    Clear the conversation history for a specific session.
    
    Args:
        session_id: Session identifier to clear
    """
    if session_id in _session_store:
        del _session_store[session_id]
        logger.info(f"Cleared session history for {session_id}")


def get_all_sessions():
    """
    Get list of all active session IDs.
    
    Returns:
        List of session IDs
    """
    return list(_session_store.keys())
