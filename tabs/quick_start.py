import streamlit as st
import os

def show():
    st.write('<p style="font-size:40px; color:cornflowerblue;">Mission</p>', unsafe_allow_html=True)
    st.markdown(
        """
        Faster creation of Superior, Compliant Claims & Visuals powered by **AI** and **SOMETHING** proprietary data.
        """
    )
    #
    # st.image('homepage.png', caption='Claim Advisor')

    st.markdown(
        """
        ## Introduction
        Claim Advisor is a GenAI powered tool that helps Beauty Care to create superior, compliant claims and visuals using AI 
        and -something- proprietary data.  

        ## Unique Advantages
        - 1)	Faster search of existing and similar claims and images
        - 2)	Generate new and improve benchmark claims based on previous consumer learnings
        - 3)	Predict consumer responses from MaxDiff to accelerate feedback cycle

        ## Data Sources 
        -   data source 1
        -   data source 2
        
        ## Demo Video
        - Here is a [demo video TBD](https://your_demo_video_link_here) for quick DIY use of Claim Advisor.

        """
    )    