import json
import requests
import streamlit as st
from groq import Groq

MODEL_NAME = "openai/gpt-oss-120b"

# ------------------------------------------------------------------------------
# 1. Weather Tool Function & Schema Definition
# ------------------------------------------------------------------------------

def get_current_weather(location: str, unit: str = "metric", weather_api_key: str = "") -> str:
    """Fetch live weather from OpenWeatherMap API."""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_api_key}&units={unit}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if response.status_code == 200:
            unit_symbol = "°C" if unit == "metric" else "°F"
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            condition = data["weather"][0]["description"]
            city = data["name"]
            country = data["sys"]["country"]
            return f"Weather in {city}, {country}: {temp}{unit_symbol} (feels like {feels_like}{unit_symbol}) with {condition}."
        elif response.status_code == 401:
            return "Error: Invalid OpenWeatherMap API key."
        elif response.status_code == 404:
            return f"Error: City '{location}' not found. Please double-check the name."
        else:
            return f"Error fetching weather: {data.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Failed to connect to weather service: {str(e)}"

# Groq tool schema for Weather Agent
weather_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather for a specific city or region.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and optional state/country, e.g., 'Lahore', 'London, UK', 'Tokyo'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["metric", "imperial"],
                        "description": "Temperature unit: 'metric' for Celsius, 'imperial' for Fahrenheit. Default is 'metric'.",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

# System Prompts for specialized agents
WEATHER_AGENT_PROMPT = """You are a STRICT Weather Specialist Agent.
Your SOLE purpose is to handle questions strictly about weather, temperature, climate, rain, or forecasts using the `get_current_weather` tool.

CRITICAL INSTRUCTIONS:
1. NEVER answer general knowledge, coding, history, definition, or non-weather questions (e.g., 'what is an LLM', 'who was Einstein', 'what is ChatGPT').
2. If the user asks ANY question that is not directly related to weather or temperature, you MUST decline immediately.
3. Your refusal message MUST be: "I am only specialized in weather forecasts and atmospheric conditions. Please switch to the General Knowledge Agent for other topics!"
"""
GK_AGENT_PROMPT = """You are a General Knowledge Specialist Agent.
Your role is to provide clear, helpful, and accurate answers to general questions covering history, science, coding, literature, trivia, and daily facts."""

# ------------------------------------------------------------------------------
# 2. Router & Agent Processing Functions
# ------------------------------------------------------------------------------

def classify_intent(client: Groq, user_prompt: str) -> str:
    """Classifies user input into 'WEATHER' or 'GENERAL_KNOWLEDGE' dynamically."""
    router_prompt = f"""Analyze the user query and decide which agent should handle it:
- "WEATHER": If the input asks about weather, temperature, rain, wind, forecasts, or climate.
- "GENERAL_KNOWLEDGE": If the input asks about general facts, history, science, coding, math, greetings, or trivia.

Respond ONLY with either "WEATHER" or "GENERAL_KNOWLEDGE".

User Query: "{user_prompt}"
Category:"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": router_prompt}],
        temperature=0.0
    )
    intent = response.choices[0].message.content.strip().upper()
    return "WEATHER" if "WEATHER" in intent else "GENERAL_KNOWLEDGE"

def handle_weather_agent(client: Groq, history: list, weather_api_key: str) -> str:
    """Handles weather queries with tool calls."""
    api_messages = [{"role": "system", "content": WEATHER_AGENT_PROMPT}] + [
        {"role": m["role"], "content": m["content"]} for m in history
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=api_messages,
        tools=weather_tools,
        tool_choice="auto",
        temperature=0.2,
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        api_messages.append(response_message)
        for tool_call in tool_calls:
            if tool_call.function.name == "get_current_weather":
                args = json.loads(tool_call.function.arguments)
                weather_result = get_current_weather(
                    location=args.get("location"),
                    unit=args.get("unit", "metric"),
                    weather_api_key=weather_api_key,
                )
                api_messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "get_current_weather",
                    "content": weather_result,
                })

        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=api_messages
        )
        return final_response.choices[0].message.content
    else:
        return response_message.content

def handle_gk_agent(client: Groq, history: list) -> str:
    """Handles general knowledge queries directly via Groq."""
    api_messages = [{"role": "system", "content": GK_AGENT_PROMPT}] + [
        {"role": m["role"], "content": m["content"]} for m in history
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=api_messages,
        temperature=0.7,
    )
    return response.choices[0].message.content

# ------------------------------------------------------------------------------
# 3. Streamlit UI & Chat Logic
# ------------------------------------------------------------------------------

st.set_page_config(page_title="Dual-Agent AI Assistant", page_icon="🤖")
st.title("🤖 Dual-Agent AI Chatbot")
st.caption("Made by M.Arhum")

# Sidebar: user-supplied API keys + agent controls
with st.sidebar:
    st.header("🔑 Your API Keys")
    st.caption(
        "Your keys are used only for your own session and are never stored or "
        "shared. Get a free Groq key at console.groq.com/keys and a free "
        "OpenWeatherMap key at openweathermap.org/api."
    )
    groq_key_input = st.text_input("Groq API Key", type="password", key="groq_key_input").strip()
    weather_key_input = st.text_input("OpenWeatherMap API Key", type="password", key="weather_key_input").strip()

    groq_format_ok = groq_key_input.startswith("gsk_") if groq_key_input else True
    weather_format_ok = len(weather_key_input) == 32 if weather_key_input else True

    if groq_key_input and not groq_format_ok:
        st.warning("Groq keys usually start with 'gsk_'. Double-check you copied it correctly.")
    if weather_key_input and not weather_format_ok:
        st.warning("OpenWeatherMap keys are usually 32 characters. Double-check you copied it correctly.")

    keys_ready = bool(groq_key_input) and bool(weather_key_input)

    st.divider()
    st.header("⚙️ Agent Controls")

    # Selection Option Bar
    selected_mode = st.selectbox(
        "Choose Target Agent:",
        [
            "⚡ Auto-Detect (Router)",
            "🌤️ Weather Agent",
            "🧠 General Knowledge Agent"
        ],
        index=0
    )

    st.divider()
    st.markdown("**Active Mode Details:**")
    if "Auto-Detect" in selected_mode:
        st.info("The Classifier LLM will automatically determine which agent to use for every prompt.")
    elif "Weather" in selected_mode:
        st.success("Forced **Weather Agent** mode (Uses OpenWeatherMap tool).")
    else:
        st.success("Forced **GK Agent** mode (General Q&A).")

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Block the chat until both keys are provided
if not keys_ready:
    st.info("👈 Enter your Groq and OpenWeatherMap API keys in the sidebar to start chatting.")
    st.stop()

# Initialize Groq client with the user's own key
client = Groq(api_key=groq_key_input)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing message history
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            if "agent" in message:
                agent_label = "🌤️ *Weather Agent*" if message["agent"] == "WEATHER" else "🧠 *General Knowledge Agent*"
                st.caption(agent_label)
            st.write(message["content"])

# User Chat Input
if prompt := st.chat_input("Ask about weather or any general topic..."):
    # Append & render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Process response based on sidebar selection
    with st.chat_message("assistant"):
        with st.spinner("Processing prompt..."):

            # Determine which agent mode to execute
            if "Weather Agent" in selected_mode:
                active_intent = "WEATHER"
            elif "General Knowledge" in selected_mode:
                active_intent = "GENERAL_KNOWLEDGE"
            else:
                # Fallback to AI Router classification
                try:
                    active_intent = classify_intent(client, prompt)
                except Exception as e:
                    st.error(f"Groq API error: {e}")
                    st.stop()

            # Execute corresponding agent
            try:
                if active_intent == "WEATHER":
                    st.caption("🌤️ *Weather Agent*")
                    final_text = handle_weather_agent(client, st.session_state.messages, weather_key_input)
                else:
                    st.caption("🧠 *General Knowledge Agent*")
                    final_text = handle_gk_agent(client, st.session_state.messages)
            except Exception as e:
                final_text = f"⚠️ Error talking to Groq API: {e}"

            st.write(final_text)

            # Store response with agent metadata
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_text,
                "agent": active_intent
            })
