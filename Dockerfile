#Base Image
FROM python:3.11-slim

#WorkDIR
WORKDIR /app 

#copy
COPY requirements.txt .

#RUN
RUN pip install -r requirements.txt

COPY . .

#PORT
EXPOSE 8501

#Command
CMD [ "streamlit","run","app.py" ]