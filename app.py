import streamlit as st
from multiapp import MultiApp
from models import home,lstm, FB_Prophet # import your app modules here

app = MultiApp()

st.markdown(""
            """# 📈📉 Welcome to STOCK PRICE PREDICTION app """)




# Add all your applications here

app.add_app("Home", home.app)
app.add_app("lstm", lstm.app)
app.add_app("fb_prophet", FB_Prophet.app)

# The main app
app.run()
