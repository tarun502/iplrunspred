FROM python:3.10.13
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD ["streamlit", "run", "app.py"]

