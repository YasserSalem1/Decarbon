import streamlit as st
import plotly.graph_objects as go

def get_figure(name):
    if name == "pie":
        fig = go.Figure(data=[go.Pie(labels=['Renewables', 'Fossils'], values=[70, 30])])
        fig.update_layout(title="Pie Chart: Energy Source Breakdown")
    elif name == "bar":
        fig = go.Figure(data=[go.Bar(x=['Solar', 'Wind', 'Fossil'], y=[35, 35, 30])])
        fig.update_layout(title="Bar Chart: Energy Contribution")
    elif name == "line":
        fig = go.Figure(data=[go.Scatter(x=list(range(8)), y=[50, 60, 70, 60, 55, 40, 45, 50], mode='lines+markers')])
        fig.update_layout(title="Line Chart: Hourly Demand")
    elif name == "gauge":
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=300,
            title={'text': "Gauge: Total Energy Supply"},
            gauge={
                'axis': {'range': [0, 400]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 200], 'color': "lightgreen"},
                    {'range': [200, 400], 'color': "lightblue"},
                ],
            }
        ))
    elif name == "stacked":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(8)), y=[20, 30, 25, 40, 35, 30, 20, 25], stackgroup='one', name='Solar'))
        fig.add_trace(go.Scatter(x=list(range(8)), y=[10, 15, 20, 25, 20, 15, 10, 15], stackgroup='one', name='Wind'))
        fig.add_trace(go.Scatter(x=list(range(8)), y=[30, 20, 15, 5, 10, 15, 20, 10], stackgroup='one', name='Fossil'))
        fig.update_layout(title="Stacked Area Chart: Energy Supply Over Time")
    else:
        fig = go.Figure()
        fig.update_layout(title="No visualization selected.")
    
    return fig

def datavisualize(chart_name="pie"):
    fig = get_figure(chart_name)
    st.plotly_chart(fig, use_container_width=True)
