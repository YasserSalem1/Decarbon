# chatbot.py

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
        return selected_chart, "I can show: pie, bar, line, gauge, stacked. Try one!"


