from utils import DEFAULT_PRODUCT_DESC, generate_claims
from tabs.max_diff_simulator import generate_result, run_simulation
import streamlit as st
import string
import random
import pandas as pd

def show():

    initializeLocalStorage()

    with st.expander("🚨Here is instruction🚨", False):
        writeInstruction()

    st.subheader("Generate Claims")
    with st.expander("Product Description"):
        opt_product_desc = st.text_area(" ", st.session_state.opt_product_description, height=200)
        st.session_state.opt_product_description = opt_product_desc

    if st.button("Generate", type="primary"):
        if(opt_product_desc==""):
            st.warning("Provide product description to generate claims")
        else:
            generateClaims([],opt_product_desc)
    
    opt_claims = []
    
    st.subheader("Optimize Claims")
    with st.expander("Claims", True):
        suggestionsPerClaim = st.slider("Number of suggested claims", min_value=0, max_value=20, value=5)#st.session_state.slider_value)
        #st.session_state.slider_value = suggestionsPerClaim

        for input_key in st.session_state.claim_opt_input_keys:
            col1, col2 = st.columns([4, 1])
            with col1:
                claim = st.text_input(" ", key=input_key, value=st.session_state.opt_claims.get(input_key, ""), label_visibility="collapsed")
                st.session_state.opt_claims[input_key] = claim
                opt_claims.append(claim)
            with col2:
                if st.button("Delete", key=f"delete_{input_key}"):
                    st.session_state.claim_opt_input_keys.remove(input_key)
                    del st.session_state.opt_claims[input_key]
                    st.rerun()

    col1, col2, col3 = st.columns([10, 2, 2])

    with col1:
        flagOptimizeButton = False
        if st.button("Optimize", type="primary"):
                st.session_state.optimized_claims = {}
                flagOptimizeButton = True
            # st.session_state.results_simulation = {} 
            
    if flagOptimizeButton:
        with st.spinner("Calculating results..."):            
            optimizeClaims(opt_claims, opt_product_desc, suggestionsPerClaim)

    with col2:
        if st.button("Add New Claim"):
            new_key = random.choice(string.ascii_uppercase) + str(random.randint(0, 999999))
            st.session_state.claim_opt_input_keys.append(new_key)
            st.session_state.opt_claims[new_key] = ""
            st.rerun()

    with col3:
        if st.button("Reset All Claim"):
            st.session_state.claim_opt_input_keys = []
            st.session_state.optimized_claims = {}
           # st.session_state.results_simulation = {}
            st.rerun()

    applyLocalStorage()


def generateClaims(opt_claims,opt_product_desc):
    st.session_state.claims = generate_claims(opt_claims, opt_product_desc)
    for claim in st.session_state.claims:
        new_key = random.choice(string.ascii_uppercase) + str(random.randint(0, 999999))
        st.session_state.claim_opt_input_keys.append(new_key)
        st.session_state.opt_claims[new_key] = claim
    st.rerun()

def checkConditionsForClaims(opt_claims):
    if not opt_claims:
        st.warning("Please provide claims")
        return False
    if [claim for claim in opt_claims if claim == ""]:
        st.warning("Empty Claims Found")
        return False
    duplicates = list(set([claim for claim in opt_claims if opt_claims.count(claim) > 1]))
    if duplicates:
        st.warning(f"Duplicate Claims Found: {duplicates}")
        return False
    if len(opt_claims) == 0:
        st.warning("Provide claims")
        return False
    return True

def optimizeClaims(opt_claims, opt_product_desc, suggestionsPerClaim):
    if checkConditionsForClaims(opt_claims):
        for claim in opt_claims:
            new_claims = generate_claims([claim], opt_product_desc, suggestionsPerClaim)
            st.session_state.optimized_claims[claim] = pd.Series(
                new_claims, 
                index=range(1, len(new_claims) + 1),
                name="Claim"
            )
        # st.subheader("Results for:")          
        # if st.session_state.optimized_claims:
        #     for source_claim, claims_series in st.session_state.optimized_claims.items():
        #         st.markdown(f"###### {source_claim}")
        #         st.table(claims_series)

def simulateClaims(total_test):
    if st.session_state.optimized_claims:
        for source_claim, claims_series in st.session_state.optimized_claims.items():
            best_counts_optimize_and_simulate, worst_count, _ = run_simulation(claims_series, st.session_state.opt_product_description, total_test)
            st.session_state.best_counts_optimize_and_simulate = best_counts_optimize_and_simulate
            st.session_state.worst_counts_optimize_and_simulate = worst_count
            result = generate_result(best_counts_optimize_and_simulate, worst_count)
            st.markdown(f"###### {source_claim}")
            st.table(result)
            #st.session_state.results_simulation[source_claim] = result

def initializeLocalStorage():
    if 'opt_product_description' not in st.session_state:
        st.session_state.opt_product_description = DEFAULT_PRODUCT_DESC.strip()

    if 'claim_opt_input_keys' not in st.session_state:
        st.session_state.claim_opt_input_keys = []

    if 'opt_claims' not in st.session_state:
        st.session_state.opt_claims = {}

    if 'optimized_claims' not in st.session_state:
        st.session_state.optimized_claims = {}

    # if 'slider_value' not in st.session_state:
    #     st.session_state.slider_value = 5

def applyLocalStorage():

    if st.session_state.optimized_claims:
        st.subheader("Results for:")  
        for source_claim, claims_series in st.session_state.optimized_claims.items():
            st.markdown(f"###### {source_claim}")
            st.table(claims_series) 

def writeInstruction():
    st.write("To generate new claims:")
    st.caption("Fill the text area for product description and click button 'Generate'. Your results will be displayed in the table 'Claims' below.")
    st.write("To optimize claims:")
    st.caption("Add your claims to the list and lick button 'Optimized'. You can also choose how many optimized claims you want to generate for every of your input claim.")
    
    caption_text = """
❗Every claim should be 20 words max.\n
Do not start by 'somebrand products' because the claim only applies to these products.\n  
Do not start by "claim: ..." because all deliverables should be claims\n❗. 
"""
    st.caption(caption_text)
    st.write("To simulate maxDiff studies on optimized claims:")
    st.caption("After you receive your optimized claims, you can decide to perform maxDiff simulation on them. If you have your own claims on witch you want to perform maxDiff sutides go to tab 'maxDiff Simulator'.")
