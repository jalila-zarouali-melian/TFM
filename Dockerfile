FROM python:3.9
# By default, listen on port 8081
EXPOSE 8080/tcp
# Set the working directory in the container
WORKDIR /app
# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install -r requirements.txt
# Copy the content of the local src directory to the working directory
COPY style.css/ .
COPY app.py .
# Specify the command to run on container start
CMD [ "python", "./app.py" ]
