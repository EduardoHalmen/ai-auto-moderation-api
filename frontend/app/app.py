import streamlit as st
import requests
import plotly.express as px

def main():

    st.title("ToxBlock")
    
    # Text input field
    user_input = st.text_input("Enter some text:")
    
    # Display the input text when submitted
    if user_input:
        response = requests.post("http://host.docker.internal:8000/evaluate_comment", json={"text": user_input})
        if response.status_code == 200:
            result = response.json()
            # Extract toxicity scores from the response
            labels = ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"]
            values = [result.get(label, 0) for label in labels]
            
            # Create a bar plot for the scores
            colors = ['#8A2BE2', '#9370DB', '#BA55D3', '#DA70D6', '#EE82EE', '#DDA0DD']
            # Create an animated bar plot using Plotly Express
            fig = px.bar(
                x=labels,
                y=values,
                color=labels,
                color_discrete_sequence=colors,
                range_y=[0.0, 1.0],
            )
            fig.update_layout(
                title='Toxicity Scores Bar Plot',
                yaxis=dict(title='Scores'),
                xaxis=dict(title='Categories'),
                bargap=0.2
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Error:", response.status_code)

if __name__ == "__main__":
    main()