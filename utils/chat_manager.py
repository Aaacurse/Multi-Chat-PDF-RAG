import json
import uuid 
from pathlib import Path
from typing import List,Dict

from langchain_classic.schema.messages import AIMessage,HumanMessage,BaseMessage

from config import STATE_FILE

class ChatManager:
    def __init__(self):
        self.pdfs:Dict={}
        self.load_state()

    def add_pdf(self,pdf_id:str,filename:str,vector_store_path:str)->str:
        first_chat_id=str(uuid.uuid4())

        self.pdfs[pdf_id]={
            'filename':filename,
            'vector_store_path':vector_store_path,
            'upload_date':str(Path(vector_store_path).stat().st_ctime),
            'chats':{
                first_chat_id:{
                    'title':'New Chat',
                    'messages':[]
                }
            }
        }
        self.save_state()
        return first_chat_id
    
    def create_chat(self,pdf_id:str,title:str="New chat")->str:
        chat_id=str(uuid.uuid4())

        self.pdfs[pdf_id]['chats']['chat_id']={
            'title':title,
            "messages":[]
        }
        self.save_state()
        return chat_id
        
    def  add_message(self,pdf_id:str,chat_id:str,message: BaseMessage):
        serialized=self._serialize_message(message)
        self.pdfs[pdf_id]['chats'][chat_id]['messages'].append(serialized)
        self.save_state()

    def get_messages(self,pdf_id:str,chat_id:str)->List[BaseMessage]:
        serialized_messages=self.pdfs[pdf_id]['chats'][chat_id]['messages']
        return [self._deserialize_message(msg) for msg in serialized_messages]
        
    def update_chat_title(self,pdf_id:str,chat_id:str,title:str):
        self.pdfs[pdf_id]['chats'][chat_id]['title']=title
        self.save_state()

    def delete_chat(self,pdf_id:str,chat_id:str):
        if chat_id in self.pdfs[pdf_id]['chats']:
            del self.pdfs[pdf_id]['chats'][chat_id]
            self.save_state()

    def delete_pdf(self,pdf_id:str):
        if pdf_id in self.pdfs:
            del self.pdfs[pdf_id]
            self.save_state()

    def _serialize_message(self,message:BaseMessage):
        return{
            'type':message.__class__.__name__,
            'content':message.content,
            'additional_kwargs':message.additional_kwargs
        }
        
    def _deserialize_message(self,data:dict)->BaseMessage:
        if data['type']=="HumanMessage":
            return HumanMessage(
                content=data['content'],
                additional_kwargs=data.get('additional_kwargs',{})
            )
        elif data['type']=="AIMessage":
            return AIMessage(
                content=data['content'],
                additional_kwargs=data.get('additional_kwargs',{})
            )
        else:
            raise ValueError(f"Unknown Message Type: {data['type']}")
        
    def save_state(self):
        STATE_FILE.write_text(json.dumps(self.pdfs,indent=2))
        
    def load_state(self):
        if STATE_FILE.exists():
            self.pdfs=json.loads(STATE_FILE.read_text())