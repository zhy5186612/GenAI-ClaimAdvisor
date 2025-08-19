# last updated on 3/13/2024
# Use a python image
FROM python:3.10 AS build

# add new user
RUN adduser --disabled-password causer
USER causer
ENV PATH="/home/causer/.local/bin:${PATH}"

#
WORKDIR /home/causer
EXPOSE your_port_number

#### Build imge and Run the container as kguser

# copy all files from current directory to work directory
RUN mkdir .streamlit
COPY --chown=causer:causer .streamlit/config.toml .streamlit
COPY --chown=causer:causer claimAdvisor2.png .
COPY --chown=causer:causer homepage.png .
COPY --chown=causer:causer requirements.txt .
COPY --chown=causer:causer ClaimAdvisorApp.py .
COPY --chown=causer:causer utils.py .
COPY --chown=causer:causer tabs/ tabs/

# copy all files from current directory to work directory
RUN pip install pip --upgrade
RUN pip install --user -r requirements.txt


CMD streamlit run ClaimAdvisorApp.py
