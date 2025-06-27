# chatbot.py

import requests
import json

API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "deepseek/deepseek-r1-0528:free"

def chat_with_bot(message):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "NetZero Optimizer",
    }
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": message + " Reply in english and in one line."}
        ],
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"API error: {e} - {response.text}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")

def handle_chat(input_text, selected_chart):
    """
    Returns updated chart type and bot response based on user input.
    """
    input_lower = input_text.lower()

    if "bar" in input_lower:
        return "bar", "Showing the Bar Chart 📊"
    elif "line" in input_lower:
        return "line", "Here's the Line Chart 📈"
    elif "gauge" in input_lower:
        return "gauge", "Gauge Chart loaded ⛽"
    elif "stacked" in input_lower or "area" in input_lower:
        return "stacked", "Stacked area chart displayed 🌞🌬⛽"
    elif "pie" in input_lower:
        return "pie", "Back to Pie Chart 🥧"
    else:
        # Use the OpenRouter API for other queries
        bot_response = chat_with_bot(input_text)
        return selected_chart, bot_response


