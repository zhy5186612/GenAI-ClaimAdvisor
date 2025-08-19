from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.embeddings import Embeddings
from azure.identity import EnvironmentCredential
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()


# Databricks Image embedding service
CLIP_IMAGE_EMBEDDING_ENDPOINT = os.getenv("CLIP_IMAGE_EMBEDDING_ENDPOINT")
CLIP_TEXT_EMBEDDING_ENDPOINT = os.getenv("CLIP_TEXT_EMBEDDING_ENDPOINT")
DATABRICKS_TOKEN=os.getenv("DATABRICKS_TOKEN")
# Data URIs
MAXDIFF = './data/toy_all_maxdiff.pkl'
CLAIM_EMBEDDING = './data/toy_all_claims.pkl'
CLAIM_LOG_EMBEDDING = './data/toy_claim_log.pkl'
IMAGE_MAXDIFF = './data/toy_img_embeddings.pkl'
CLAIM_RANKS = './data/toy_embedding.pkl'
#
# MAXDIFF = os.getenv("ALL_MAXDIFF")
# CLAIM_EMBEDDING = os.getenv("ALL_CLAIM")
# CLAIM_LOG_EMBEDDING = os.getenv("CLAIM_LOG")
# IMAGE_MAXDIFF = os.getenv("IMAGE_MAXDIFF")
# CLAIM_RANKS = os.getenv('CLAIM_RANKS')
# P&G Azure OAI credentials 
GENAI_PROXY = os.getenv('GENAI_PROXY')
CONGNITIVE_SERVICES = os.getenv('CONGNITIVE_SERVICES')
OPEN_API_VERSION = os.getenv('OPEN_API_VERSION')
LLM_DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME')
EMBEDDING_DEPLOYMENT_NAME = os.getenv('EMBEDDING_DEPLOYMENT_NAME')
MAXDIFF_SIM_DEPLOYMENT_NAME = os.getenv('MAXDIFF_SIM_DEPLOYMENT_NAME')
if(not MAXDIFF_SIM_DEPLOYMENT_NAME):
    MAXDIFF_SIM_DEPLOYMENT_NAME = LLM_DEPLOYMENT_NAME
    
HEADERS = {
    "userid": "--",
    "project-name": "--",
}

DEFAULT_MAXDIFF_MESSAGE="""
You are a woman, ages 18 to 55 , picking skin care product based on the claim of the product.
You are primarily concerned with fine lines and wrinkles, pore size, uneven texture and tone and dark spots

Here's some information about yourself:     
    Emotional: GREATER GOOD, SELF-PROMOTION, ENJOYMENT  
    Life: Get Married, Move Careers, Move Homes, Have Kids, Kids Graduate, Kids Move Away, Grandkids, Go through Divorce, Stabilize Career, Menopause  
    Products: acne, facial cleanser, facial moisturizer, SPF, serum, eye cream, device, procedure  
    Skin needs: CLEAN, REPAIR, ENHANCE, PREVENT  
    Loss of Elasticity, Marks, Signs of Aging, Uneven or Dull Skin, Combination Skin, Unpredictable Skin, Age Spots, Brown Spots, Dark Spots, Dullness, Sunspots, Aging around Mouth, hands, neck, Visible Pores, Uneven Skin Texture  
    Brands: Clinique, Clarin, Drunk Elephant, Kielh’s, Garnier, Estee Lauder, Aveeon, Lancome, Loreal, Olay, Simple, Tatcha, Tula, Neutrogena   
    Stores: DEPARTMENT, MASS, DRUG, FOOD, ONLINE, SPECIALTY, ONLINE, DERM, SPA, CATALOGUE  
    Media: facebook, Instagram, google, DTC  

You are looking for these BENEFITS:  
    Undo damage from the prior you  
    Stretch out time needed for invasive procedures  
    Over the counter face lift  
    Reveal the next best layer of skin, self.  
    Deepest penetrating ingredients for the highest amount of efficacy.  
    acute and chronic benefits of the product 
    noticeable difference in the skin 
"""

DEFAULT_PRODUCT_DESC="""
For example:

The product is targeted for women ages 18 to 55 who are primarily concerned with fine lines and wrinkles, pore size, uneven texture and tone and dark spots.  

Some product characteristics that your target consumer cares when buying a product:  
    Light Feel 
    Moisture with some skin coverage 
    Hydration 
    Healthy looking hydration 

"""

CLAIM_OPTIMIZATION_EXAMPLE_RESPONSE = [
# Performace Based
    "Sleep is essential to every organ in the body, including your skin",
    "82% Visible Improvement in 7 Days with Olay",
    "10 Benefits in 1 jar: Moisturize, Firm, Smooth Wrinkles, Lift, Regenerate, Strengthen Moisture Barrier, Brighten, Nourish, Plump, and Even Tone",
    "Skin Renewal Collection",
    "Hydrates and renews for smooth, bright skin",
    "Repairs signs of Aging",
    "Minimizes the Look of Fine Lines and Wrinkles",
# Semantic Based
    "Sleep is essential to every organ in the body, including your skin",
    "82% Visible Improvement in 7 Days with Olay",
    "10 Benefits in 1 jar: Moisturize, Firm, Smooth Wrinkles, Lift, Regenerate, Strengthen Moisture Barrier, Brighten, Nourish, Plump, and Even Tone",
    "Skin Renewal Collection",
    "Hydrates and renews for smooth, bright skin",
    "Repairs signs of Aging",
    "Minimizes the Look of Fine Lines and Wrinkles",
]

CLAIM_OPTIMIZATION_EXAMPLE = [
# Performance Based
    """
    Here's some claims with good performance:
    Claim: Your skin naturally changes throughout your life due to biological changes & other stress
    Performance: 1.253720664235294
    Claim: When you have good skin, you're more confident
    Performance: 1.0693704075294117
    Claim: Night time is the easiest way to invest 8 hours of care into your skin - while you sleep!
    Performance: 1.0607506291764706
    Claim: Repairing your skin is great, but what if you could repair it and protect from future damage
    Performance: 1.0282695
    Claim: You don't have to pay luxury prices for a good moisturizer
    Performance: 0.962742659764706
    Read claims above and create a new similar claim makes our target consumer more likely to purchase the product.
    """,
# Semantic Based
    """
    Here's some claims:
    Getting your “beauty” sleep is real because your skin renews and repairs at night
    Not getting enough sleep shows up the most on your skin, in the morning
    Your skin repairs itself when you are asleep
    Your skin is your body's first line of defense against everyday stressors
    Night time is the easiest way to invest 8 hours of care into your skin - while you sleep!
    Read claims above and create a new similar claim makes our target consumer more likely to purchase the product.
    """,
]

@st.cache_data
def load_images()->pd.DataFrame:
    return pd.read_pickle(IMAGE_MAXDIFF)

@st.cache_data
def load_claim_log_embeddings()->pd.DataFrame:
    claim_log = pd.read_pickle(CLAIM_LOG_EMBEDDING)
    claim_log["Claim"] = claim_log.Claim.apply(str.strip)
    claim_log["Product Line"] = claim_log["Product Line"].apply(str.strip)
    claim_log = claim_log[claim_log.Claim.apply(lambda x: x!="")]
    claim_log.drop_duplicates(inplace=True, subset=["Product Line", "Product", "Claim"])
    return claim_log

@st.cache_data
def load_maxdiff()->pd.DataFrame:
    # storage_options = {'User-Agent': 'Mozilla/5.0'}
    maxdiff = pd.read_pickle(MAXDIFF)
    return maxdiff

@st.cache_data
def load_claim_embeddings()->pd.DataFrame:
    claims = pd.read_pickle(CLAIM_EMBEDDING)
    claims["claim"] = claims.claim.apply(str.strip)
    claims = claims[claims.claim.apply(lambda x: x!="")]
    return claims

@st.cache_data
def load_claim_ranks()->pd.DataFrame:
    claims :pd.DataFrame =  pd.read_pickle(CLAIM_RANKS)
    claims["claim"] = claims.claim.apply(lambda x: x.strip())
    claims.drop("embedding", axis=1, inplace=True)
    return claims

def load_data():
    maxdiffs = load_maxdiff()
    claim_embeddings = load_claim_embeddings()
    claim_log = load_claim_log_embeddings()
    return maxdiffs, claim_embeddings, claim_log

def text_to_embedding(text:str)->list:
    def get_text_embedding_model()->Embeddings:
        # Azure OpenAI The context window of gpt-4 is 32k tokens
        env_credential = EnvironmentCredential()
        token = env_credential.get_token(CONGNITIVE_SERVICES).token
        model = AzureOpenAIEmbeddings(
            azure_endpoint=GENAI_PROXY,
            deployment=EMBEDDING_DEPLOYMENT_NAME,
            api_version=OPEN_API_VERSION,
            api_key=token,
            default_headers=HEADERS
        )
        return model
    if("embedding_model" not in st.session_state):
        st.session_state.embedding_model = get_text_embedding_model()
    try:
        res = st.session_state.embedding_model.embed_query(text)
    except:
        st.warning("Something goes wrong! Requesting new token.")
        st.session_state.embedding_model = get_text_embedding_model()
        res = st.session_state.embedding_model.embed_query(text)
    return res

def text_to_clip_embedding(text:str)->np.ndarray:
    data_json = json.dumps({
        "dataframe_split": {
            "columns": ["text"],
            "data": [[text]]
    }}, allow_nan=True)
    headers = {'Authorization': f'Bearer {DATABRICKS_TOKEN}', 'Content-Type': 'application/json'}
    response = requests.request(method='POST', headers=headers, url=CLIP_TEXT_EMBEDDING_ENDPOINT, data=data_json)
    res = response.json()["predictions"][0]
    return np.array(res)

def imageb64_to_clip_embedding(image_b64:str)->np.ndarray:
    data_json = json.dumps({
        "dataframe_split": {
            "columns": ["img"],
            "data": [[image_b64]]
    }}, allow_nan=True)
    headers = {'Authorization': f'Bearer {DATABRICKS_TOKEN}', 'Content-Type': 'application/json'}
    response = requests.request(method='POST', headers=headers, url=CLIP_IMAGE_EMBEDDING_ENDPOINT, data=data_json)
    res = response.json()["predictions"][0]
    return np.array(res)

def get_maxdiff_example(system_message:str=None):
    df = load_claim_ranks()
    if(system_message):
        example_messages = [SystemMessage(content=system_message)]
    else:
        example_messages = [SystemMessage(content=DEFAULT_MAXDIFF_MESSAGE)]
    train_research = df.research_name.drop_duplicates()
    for _ in range(300):
        random_research = train_research.sample(1)[0]
        research_df = df.loc[df.research_name==random_research]
        # random sample from claims
        sampled_claim = research_df.sample(5).reset_index(drop=True)
        # get the index of best/worst performed claim 
        sorted_sample_claim = sampled_claim.sort_values("maxdiff")
        best = sorted_sample_claim.iloc[-1].name 
        worst = sorted_sample_claim.iloc[0].name
        claims = "Pick the best and worst claim:\n"
        for idx, c in enumerate(sampled_claim.claim):
            claims += f"{idx}. {c}\n"
        example_messages.append(HumanMessage(content=claims))
        example_messages.append(AIMessage(content=f"Best:{best}, Worst:{worst}"))
    return example_messages

def get_chat_client()->AzureChatOpenAI:
    env_credential = EnvironmentCredential()
    token = env_credential.get_token(CONGNITIVE_SERVICES).token
    MAXDIFF_SIM_DEPLOYMENT_NAME
    chat = AzureChatOpenAI(
        azure_endpoint= GENAI_PROXY,
        azure_deployment=MAXDIFF_SIM_DEPLOYMENT_NAME,
        api_version= OPEN_API_VERSION,
        api_key=token,
        temperature=0,
        default_headers=HEADERS,
    )
    return chat 

def generate_claims(claims, product_desc, num_claims_per_element=1):
    messages = []
    # remove empty claims
    claims = pd.Series(claims).dropna()
    claims = claims.loc[claims.str.strip() != ""]
    
    # Create system message
    system_message = "You work in the marketing department of Olay skin care brand.\n"
    system_message += "You need to create and improve product claims for our target consumer.\n"
    system_message += DEFAULT_PRODUCT_DESC
    messages.append(SystemMessage(content=system_message))
    
    # Create examples
    for h, a in zip(CLAIM_OPTIMIZATION_EXAMPLE, CLAIM_OPTIMIZATION_EXAMPLE_RESPONSE):
        messages.append(HumanMessage(content=h))
        messages.append(AIMessage(content=a))
    
    # Add new system message
    system_message = "You work in the marketing department of Olay skin care brand.\n"
    system_message += "You need to create and improve product claims for our target consumer.\n"
    system_message += product_desc
    messages.append(SystemMessage(content=system_message))
    
    new_claims = []
    chat = get_chat_client()
    
    if len(claims) == 0:
        # Generate claims based on product_desc if claims list is empty
        human_message = "Generate some claims to make our target consumer more likely to purchase the product.\n"
        human_message += "Output Format:\n"
        human_message += "{id}. {new claim}\n"
        messages.append(HumanMessage(human_message))
        
        res = chat(messages).content.split("\n")
        for c in res:
            if c.strip() != "":
                new_claims.append(c.split(".")[1].strip())
    else:
        for claim in claims:
            # Create human message for each claim
            human_message = f"Read the following claim and generate {num_claims_per_element} new claims to make our target consumer more likely to purchase the product.\n"
            human_message += f"1. {claim}\n"
            # Set output format
            human_message += "Output Format:\n"
            human_message += "{id}. {new claim}\n"
            messages.append(HumanMessage(human_message))
            
            # send API request
            claim_new_claims = []
            while len(claim_new_claims) < num_claims_per_element:
                res = chat(messages).content.split("\n")
                for c in res:
                    if c.strip() != "":
                        claim_new_claims.append(c.split(".")[1].strip())
                claim_new_claims = claim_new_claims[:num_claims_per_element]  # Ensure we don't exceed the desired number of claims
            
            new_claims.extend(claim_new_claims)
            messages.pop()  # Remove the last human message to prepare for the next claim
    
    return new_claims
