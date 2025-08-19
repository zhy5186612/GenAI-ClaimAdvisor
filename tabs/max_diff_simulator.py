from langchain.schema import HumanMessage
from utils import get_maxdiff_example, get_chat_client, DEFAULT_MAXDIFF_MESSAGE
import streamlit as st
import pandas as pd
import random
import string

def run_simulation(new_claims, consumer_desc=None, run_times=100):
    progress_bar = st.progress(0, text=f"Progress: 0/{run_times}")
    new_claims = pd.Series(new_claims).dropna()
    new_claims = new_claims.loc[new_claims.str.strip()!=""]
    chat = get_chat_client()
    best_count_dict = {}
    worst_count_dict = {}
    crash=0
    example_messages = get_maxdiff_example(consumer_desc)
    for i in range(run_times):
        messages = example_messages.copy()
        sampled_claim = new_claims.sample(5).reset_index(drop=True)
        claims = "Pick the best and worst claim:\n"
        for idx, c in enumerate(sampled_claim):
            claims += f"{idx}. {c}\n"
        messages.append(HumanMessage(content=claims))
        pred = chat(messages).content
        try:
            pred_best = int(pred.split(",")[0][-1])
            pred_worst = int(pred[-1])
            # Best claim selected count
            pred_best_claim = sampled_claim.iloc[pred_best]
            if pred_best_claim in best_count_dict.keys():
                best_count_dict[pred_best_claim] += 1
            else:
                best_count_dict[pred_best_claim] = 1
            # Worst claim selected count
            pred_worst_claim = sampled_claim.iloc[pred_worst]
            if pred_worst_claim in worst_count_dict.keys():
                worst_count_dict[pred_worst_claim] += 1
            else:
                worst_count_dict[pred_worst_claim] = 1
        except Exception as e:
            crash+=1
            print(f"Error processing prediction: {e}")
        del messages
        progress_bar.progress((i+1)/run_times, text=f"Progress: {i+1}/{run_times}")

    progress_bar.empty()
    return best_count_dict, worst_count_dict, crash

def generate_result(best_count_max_diff_simulator, worst_counts_max_diff_simulator):
    best_count_max_diff_simulator = pd.Series(best_count_max_diff_simulator, name="best_count").sort_values(ascending=False)
    worst_counts_max_diff_simulator = pd.Series(worst_counts_max_diff_simulator, name="worst_count").sort_values(ascending=False)
    pred = pd.DataFrame([best_count_max_diff_simulator, worst_counts_max_diff_simulator]).T
    pred.index.name="Claim"
    pred.fillna(0, inplace=True)
    pred["Pick Rate"] = pred.best_count/(pred.best_count+pred.worst_count)
    pred.sort_values(["best_count", "Pick Rate", "worst_count"], ascending=[False, False, True])
    pred.columns = ["Best Count", "Worst Count", "Best Rate"]
    # st.table(pred[["Pick Rate"]])
    pred.reset_index(inplace=True) 
  
    return pred

def show():
    flag = False
    total_test = st.sidebar.slider("Simulation Repeats", 50, 500)

    st.caption("🚨 Enter at least 5 claims to simulate MaxDiff study. You can also add consumer description for better results") 

    if 'input_keys' not in st.session_state:
        st.session_state.input_keys = []

    if 'maxDiffClaims' not in st.session_state:
        st.session_state.maxDiffClaims = {}

    if 'consumer_description' not in st.session_state:
        st.session_state.consumer_description = DEFAULT_MAXDIFF_MESSAGE.strip()

    with st.expander("Consumer Description"):
        consumer_desc = st.text_area(" ", st.session_state.consumer_description, height=200)
        st.session_state.consumer_description = consumer_desc

    claims = []
    with st.expander("Claims", True):
        for input_key in st.session_state.input_keys:
            col1, col2 = st.columns([4, 1])
            with col1:
                claim = st.text_input(" ", key=input_key, value=st.session_state.maxDiffClaims.get(input_key, ""), label_visibility="collapsed")
                st.session_state.maxDiffClaims[input_key] = claim
                claims.append(claim)
            with col2:
                if st.button("Delete", key=f"delete_{input_key}"):
                    st.session_state.input_keys.remove(input_key)
                    del st.session_state.maxDiffClaims[input_key]
                    st.rerun()

    col1, _, col2, col3 = st.columns([2, 6, 2, 2])
    with col1:
        if st.button("Add New Claim", type="primary"):
            new_key = random.choice(string.ascii_uppercase) + str(random.randint(0, 999999))
            st.session_state.input_keys.append(new_key)
            st.session_state.maxDiffClaims[new_key] = ""
            st.rerun()
    with col2:
        if st.button("Submit"):
            flag = True
    if flag:
        if len(claims) < 5:
            st.warning("Please Give More Than 5 Claims")
            flag = False

        if [claim for claim in claims if claim == ""]:
            st.warning("Empty Claims Found")
            flag = False

        duplicates = list(set([claim for claim in claims if claims.count(claim) > 1]))
        if duplicates:
            st.warning(f"Duplicate Claims Found: {duplicates}")
            flag = False

    with col3:
        if st.button("Reset All Claim"):
            st.session_state.input_keys = []
            st.session_state.maxDiffClaims = {}
            st.rerun()

    if flag:
        best_count_max_diff_simulator, worst_count, _ = run_simulation(claims, consumer_desc, total_test)
        st.session_state.best_count_max_diff_simulator = best_count_max_diff_simulator
        st.session_state.worst_counts_max_diff_simulator = worst_count

    if 'best_count_max_diff_simulator' in st.session_state and 'worst_counts_max_diff_simulator' in st.session_state:
        st.subheader("Results")
        result = generate_result(st.session_state.best_count_max_diff_simulator, st.session_state.worst_counts_max_diff_simulator)
        st.table(result)
