FROM python:3.12-slim

# update the package list and install the necessary dependencies for building 
# Python packages and PostgreSQL client libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# upgrade pip to the latest version
RUN pip install --upgrade pip

# create a folder named code in the image and copy all the files to it
COPY . /code

# make the code folder executable
RUN chmod +x /code

# install the dependencies from the requirements.txt file
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# expose the port 8005 to the outside world
EXPOSE 8005

# set the working directory to code
WORKDIR /code

CMD ["python", "main.py"]
