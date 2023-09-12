FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Install Streamlit within the container
RUN pip install streamlit

copy . .

ENTRYPOINT ["streamlit", "run", "app.py"]
##cmd ["python3", "app.py"]