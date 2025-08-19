from utils import text_to_embedding, load_maxdiff, load_claim_embeddings, load_claim_log_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import streamlit as st
import pandas as pd

PRIORITIZED_CLAIM_LOG_FILTER = [
    "brand 1", 
    "brand 2",
    "brand 3",   
    "brand 4",
    "brand 5", 
]
# Apply filters

def apply_claim_log_filter(filter:dict, claim_log:pd.DataFrame)->pd.DataFrame:
    for product, checked in filter.items():
        if(not checked):
            claim_log = claim_log.loc[
                claim_log["Product Line"] != product]
    claim_log = claim_log if len(claim_log) else load_claim_log_embeddings()
    return claim_log

def apply_maxdiff_tester_filters(filters: List[dict], maxdiffs: pd.DataFrame) -> pd.DataFrame:
    original_maxdiffs = maxdiffs.copy()
    for f in filters:
        if not f:
            continue
        mask = pd.Series([False] * len(maxdiffs), index=maxdiffs.index)
        for e, checked in f.items():
            mask = mask | maxdiffs[e] if checked else mask
        maxdiffs = maxdiffs.loc[mask]
    return maxdiffs if not maxdiffs.empty else original_maxdiffs

def apply_maxdiff_age_filter(age_from:int, age_to:int, maxdiffs:pd.DataFrame)->pd.DataFrame:
    maxdiffs = maxdiffs.loc[(maxdiffs.Age>=age_from) & (maxdiffs.Age<=age_to)]
    return maxdiffs

def apply_maxdiff_research_filter(filter:dict, claim_embeddings:pd.DataFrame)->pd.DataFrame:
    if(not filter):
        return claim_embeddings
    
    for research, checked in filter.items():
        if(not checked):
            claim_embeddings = claim_embeddings.loc[
                claim_embeddings.research_name != research]
    claim_embeddings = claim_embeddings if len(claim_embeddings) else load_claim_embeddings()
    return claim_embeddings

# Show items

def show_query_input()->str:

    if 'query' not in st.session_state:
        st.session_state.query = ""

    with st.form(key='query_form'):
        query = st.text_input("Type some keywords for your claim", value=st.session_state.query)
        st.session_state.query = query

        submit_button = st.form_submit_button(label='Search')
    return query, submit_button

def show_claim_log_filter(claim_log:pd.DataFrame)->dict:
    filter = {}
    if(st.sidebar.toggle("Product Line")):
        product_lines = list(claim_log["Product Line"].unique())
        for p in PRIORITIZED_CLAIM_LOG_FILTER:
            filter[p] = st.sidebar.checkbox(f"{p}", False)
            product_lines.remove(p)

        for p in product_lines:
            filter[p] = st.sidebar.checkbox(f"{p}", False)
    return filter

def show_maxdiff_tester_filter(maxdiffs:pd.DataFrame, filter_name:str, filter_column_prefix:str)->dict:
    maxdiff_filter = {}
    if(st.sidebar.toggle(filter_name, key=filter_name)):
        for e in maxdiffs.columns:
            if(e.startswith(filter_column_prefix)):
                maxdiff_filter[e] = st.sidebar.checkbox(e[len(filter_column_prefix):], False, key=e)
    return maxdiff_filter

def show_maxdiff_age_filter(maxdiffs:pd.DataFrame)->Tuple[int, int]:
    age_max = maxdiffs.Age.max()
    age_min = maxdiffs.Age.min()
    if(st.sidebar.toggle("Age")):
        age_from, age_to = st.sidebar.slider("Age Filter", age_min, age_max, (age_min, age_max))
        return age_from, age_to
    else:
        return age_min, age_max

def show_maxdiff_research_filter(claim_embeddings:pd.DataFrame)->dict:
    filter = {}
    if(st.sidebar.toggle("Research")):
        for r in claim_embeddings.research_name.unique():
            filter[r] = st.sidebar.checkbox(r, False)
    return filter

# MaxDiff Similarity

def calculate_claim_similarity(claim_embeddings:pd.DataFrame, query:str):
    embedding = text_to_embedding(query)
    claim_embeddings["similarity"] = claim_embeddings.apply(
                lambda r: cosine_similarity([r["embedding"], embedding])[0][1], axis=1)
    claim_embeddings = claim_embeddings.sort_values("similarity", ascending=False)
    return claim_embeddings

def calculate_maxdiff(maxdiff:pd.DataFrame, claim:pd.Series)->float:
    return maxdiff.loc[maxdiff.research_name==claim.research_name][claim.id].mean()

def get_maxdiff_from_claims(maxdiff:pd.DataFrame, claims:pd.DataFrame):
    claims["maxdiff"] = claims.apply(lambda c: calculate_maxdiff(maxdiff, c), axis=1)
    claims.dropna(inplace=True)
    claims = claims[["claim", "similarity", "maxdiff", "research_name"]]
    claims.columns = ["Claim", "Similarity", "MaxDiff", "Test"]
    return claims

def get_similar_maxdiff_claims(maxdiffs:pd.DataFrame, claim_embeddings:pd.DataFrame, query:str)->Tuple[pd.DataFrame, float]:
    claim_embeddings=calculate_claim_similarity(claim_embeddings, query)
    similar_claims_maxdiff = get_maxdiff_from_claims(
        maxdiffs, claim_embeddings
    )
    cos_sim_avg = similar_claims_maxdiff[["Similarity"]].mean().values[0]
    return similar_claims_maxdiff, cos_sim_avg

# Claim Log Similarity

def calculate_claim_log_similarity(claim_log:pd.DataFrame, query:str)->pd.DataFrame:
    embedding = text_to_embedding(query)
    claim_log["similarity"] = claim_log.apply(
                lambda r: cosine_similarity([r["embedding"], embedding])[0][1], axis=1)
    claim_log = claim_log.sort_values(
        "similarity", ascending=False
    ).drop(
        ["embedding"], axis=1
    )
    return claim_log  

def show():
    maxdiffs = load_maxdiff()
    claim_embeddings = load_claim_embeddings()
    claim_log = load_claim_log_embeddings()
    query, submit_button = show_query_input()
    num_results = st.sidebar.slider("Number of results to display", 1, 50, 5)
    st.sidebar.markdown("### Claim Log Filters")
    claim_log_filter = show_claim_log_filter(claim_log)
    st.sidebar.markdown("### MaxDiff Filters")
    research_filter = show_maxdiff_research_filter(claim_embeddings)
    age_from, age_to = show_maxdiff_age_filter(maxdiffs)
    maxdiff_filters = [
        show_maxdiff_tester_filter(maxdiffs, "Ethnicity", "Ethnicity-"),
        show_maxdiff_tester_filter(maxdiffs, "Products Types Used", "ProductsTypesUsed-"),
        show_maxdiff_tester_filter(maxdiffs, "Brand Used", "BrandUsed-"),
        show_maxdiff_tester_filter(maxdiffs, "Shopping", "Shopping-"),
        show_maxdiff_tester_filter(maxdiffs, "Skin Goals", "SkinGoals-"),
        show_maxdiff_tester_filter(maxdiffs, "Benefits Sought", "BenefitsSought-"),
    ]

    if submit_button:
        if not query:
            st.warning("Please input some keywords")
            st.stop()
        
        with st.spinner("Calculating results..."):
            claim_embeddings = apply_maxdiff_research_filter(research_filter, claim_embeddings)
            maxdiffs = apply_maxdiff_age_filter(age_from, age_to, maxdiffs)
            maxdiffs = apply_maxdiff_tester_filters(maxdiff_filters, maxdiffs)
            claim_log = apply_claim_log_filter(claim_log_filter, claim_log)
            
            if query:
                maxdiff_sims, claim_log_sims = st.columns(2)
                with maxdiff_sims:
                    similar_claims_maxdiff, cos_sim_avg = get_similar_maxdiff_claims(maxdiffs, claim_embeddings, query)
                    similar_claims_maxdiff = similar_claims_maxdiff.iloc[:num_results]
                    similar_claims_maxdiff.sort_values("MaxDiff", ascending=False, inplace=True)
                    similar_claims_maxdiff.reset_index(drop=True, inplace=True)
                    similar_claims_maxdiff.index += 1
                    st.session_state.similar_claims_maxdiff = similar_claims_maxdiff
                    st.session_state.cos_sim_avg = cos_sim_avg
                with claim_log_sims:
                    if query:
                        claim_log = calculate_claim_log_similarity(claim_log, query)
                        claim_log_sim_avg = claim_log[['similarity']].mean().values[0]
                        claim_log.sort_values("similarity", ascending=False, inplace=True)
                    else:
                        claim_log.drop(["embedding"], axis=1, inplace=True)
                        claim_log_sim_avg = 0
                    st.session_state.claim_log = claim_log
                    st.session_state.claim_log_sim_avg = claim_log_sim_avg
            else:
                claim_embeddings["maxdiff"] = claim_embeddings.apply(lambda c: calculate_maxdiff(maxdiffs, c), axis=1)
                similar_claims_maxdiff = claim_embeddings[["claim", "maxdiff", "research_name"]]
                similar_claims_maxdiff.columns = ["Claim", "MaxDiff", "Test"]
                similar_claims_maxdiff.sort_values("MaxDiff", ascending=False, inplace=True)
                similar_claims_maxdiff.dropna(inplace=True)
                result_len = len(similar_claims_maxdiff)
                similar_claims_maxdiff = similar_claims_maxdiff.iloc[:num_results]
                similar_claims_maxdiff.reset_index(drop=True, inplace=True)
                similar_claims_maxdiff.index += 1
                st.session_state.similar_claims_maxdiff = similar_claims_maxdiff
                st.session_state.result_len = result_len

    if 'similar_claims_maxdiff' in st.session_state:
        st.write("Most similar claims from MaxDiff:")
        st.markdown(f"- Found {len(st.session_state.similar_claims_maxdiff)} claims, with {len(maxdiffs)} tester.")
        st.markdown(f"- Average Claim Similarity:{st.session_state.cos_sim_avg:.2f}")
        st.table(st.session_state.similar_claims_maxdiff)

    if 'claim_log' in st.session_state:
        st.write("Most similar claims from log:")
        st.markdown(f"- Found {len(st.session_state.claim_log)} Records")
        st.markdown(f"- Average Similarity:{st.session_state.claim_log_sim_avg:.2f}")
        claim_log = st.session_state.claim_log.iloc[:num_results]
        claim_log.reset_index(drop=True, inplace=True)
        claim_log.index += 1
        st.table(claim_log)