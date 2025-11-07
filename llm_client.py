from typing import Dict, List, Optional
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are Mission Control, a NASA mission expert who answers questions "
    "about Apollo and Challenger missions. Base every answer on the retrieved "
    "context provided to you. Cite the most relevant snippet IDs or sources in "
    "square brackets (e.g., [1], [AS13_TEC_excerpt]) so readers can trace the "
    "information. If the context does not contain the answer, state that "
    "clearly and offer to look for more information. Never fabricate details "
    "beyond the retrieved context."
)

def build_message_history(
    user_message: str,
    context: str,
    conversation_history: Optional[List[Dict]] = None,
    max_history_messages: int = 8,
) -> List[Dict]:
    """Assemble the message list for the OpenAI chat completion."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context:
        context_instructions = (
            "Retrieved context:\n"
            f"{context}\n\n"
            "Use only this information when responding. Cite the chunk number or "
            "source name that appears in brackets within the context."
        )
        messages.append({"role": "system", "content": context_instructions})
    else:
        messages.append({
            "role": "system",
            "content": (
                "No retrieved context is available for this question. "
                "Explain to the user that you cannot provide a grounded answer "
                "and suggest running the retrieval step again."
            )
        })

    if conversation_history:
        trimmed_history = conversation_history[-max_history_messages:]
        for entry in trimmed_history:
            role = entry.get("role")
            content = entry.get("content")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""
    if not openai_key:
        return "Error: Missing OpenAI API key."

    messages = build_message_history(
        user_message=user_message,
        context=context,
        conversation_history=conversation_history or []
    )

    try:
        client = OpenAI(api_key=openai_key)
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        return f"Error generating response: {exc}"
