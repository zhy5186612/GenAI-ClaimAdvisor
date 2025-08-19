from utils import DEFAULT_PRODUCT_DESC, generate_claims
import streamlit as st
import pandas as pd
import string
import random    

def show():
    flag = False

    st.caption("🚨 Enter a product description to generate your claims. Enter claims to optimize them. For the best results provide both.") 
   
    if 'product_description' not in st.session_state:
        st.session_state.product_description = DEFAULT_PRODUCT_DESC.strip()

    with st.expander("Product Description"):
        product_desc = st.text_area(" ", st.session_state.product_description, height=200)
        st.session_state.product_description = product_desc
    
    if 'claim_gen_input_keys' not in st.session_state:
        st.session_state.claim_gen_input_keys = []

    if 'generated_claims' not in st.session_state:
        st.session_state.generated_claims = []

    if 'gen_claims' not in st.session_state:
        st.session_state.gen_claims = {}
    
    gen_claims = []
    generated_claims = []

    expander_text = """
Every claim should be 20 words max.\n
Do not start by 'somebrand products' because the claim only applies to these products.\n  
Do not start by "claim: ..." because all deliverables should be claims\n. 
"""

    with st.expander("Claims to optimize", True):
        st.caption(expander_text)
        for input_key in st.session_state.claim_gen_input_keys:
            col1, col2 = st.columns([4, 1])
            with col1:
                claim = st.text_input(" ", key=input_key, value=st.session_state.gen_claims.get(input_key, ""), label_visibility="collapsed")
                st.session_state.gen_claims[input_key] = claim
                gen_claims.append(claim)
            with col2:
                if st.button("Delete", key=f"delete_{input_key}"):
                    st.session_state.claim_gen_input_keys.remove(input_key)
                    del st.session_state.gen_claims[input_key]
                    st.rerun()

    col1,_, col2,col3 = st.columns([2, 6, 2, 2])
    with col1:
        if st.button("Add New Claim", type="primary"):
            new_key = random.choice(string.ascii_uppercase) + str(random.randint(0, 999999))
            st.session_state.claim_gen_input_keys.append(new_key)
            st.session_state.gen_claims[new_key] = ""
            st.rerun()
    with col2:
        if st.button("Submit"):
            flag=True
    
    if flag:
        if[claim for claim in gen_claims if claim==""]:
            st.warning("Empty Claims Found")
            flag=False

        duplicates = list(set([claim for claim in gen_claims if gen_claims.count(claim) > 1]))
        if(duplicates):
            st.warning(f"Duplicate Claims Found: {duplicates}")
            flag=False

        if(product_desc=="" and len(gen_claims)==0):
            st.warning("Provide claims or product description")
            flag=False

    with col3:
        if st.button("Reset All Claim"):
            st.session_state.claim_gen_input_keys = []
            st.session_state.generated_claims = []
            st.rerun()
        
    if flag:

        with st.spinner("Calculating results..."):
            generated_claims = generate_claims(gen_claims, product_desc)
            # set index start from 1
            generated_claims = pd.Series(
                generated_claims, 
                index=range(1,len(generated_claims)+1),
                name = "Claim"
            )

        st.session_state.generated_claims = generated_claims

    if(len(st.session_state.generated_claims)!=0):
        st.table(st.session_state.generated_claims)
