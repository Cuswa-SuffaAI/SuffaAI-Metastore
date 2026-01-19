# pull the official base image
FROM python:3.11-bullseye

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN pip install --upgrade pip 

COPY ./requirements.txt /usr/src/app

RUN pip install -r requirements.txt
RUN apt update
RUN apt-get install software-properties-common -y 

# copy project
COPY . /usr/src/app

EXPOSE 7000

CMD ["python3", "manage.py", "runserver", "0.0.0.0:7000"]