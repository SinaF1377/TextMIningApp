import nltk
from nltk.stem import WordNetLemmatizer
lemsatizer = WordNetLemmatizer()
import pickle
import numpy as np
import os
from hazm import word_tokenize
import nltk
from hazm.stemmer import Stemmer
stemmer = Stemmer()

current_dir = os.path.dirname(os.path.realpath(__file__))
model = os.path.join(current_dir, 'models', 'News.jsdh')
News = open(model, 'rb')
v = pickle.load(News)
News.close()

model = os.path.join(current_dir, 'models', 'Label.jsdh')
Label = open(model, 'rb')
le = pickle.load(Label)
Label.close()

model = os.path.join(current_dir, 'models', 'svc.jsdh')
svc = open(model, 'rb')
svc_ = pickle.load(svc)
svc.close()

model = os.path.join(current_dir, 'stopwords.txt')
with open(model, encoding='utf8') as stopwords_file:
    stopwords = stopwords_file.readlines()
stopwords = [str(line).replace('\n', '') for line in stopwords]


import tkinter
from tkinter import *

def predict_class(news):
    title_body_tokenized = word_tokenize(news)
    title_body_tokenized_filtered = [w for w in title_body_tokenized if not w in stopwords]
    title_body_tokenized_filtered_stemming = [stemmer.stem(w) for w in title_body_tokenized_filtered]
    x = [' '.join(title_body_tokenized_filtered_stemming)]
    x_v = v.transform(x)
    p = svc_.predict(x_v)
    label = le.inverse_transform(p)
    return label[0]

def _onKeyRelease(event):
    ctrl = (event.state & 0x4) != 0
    if event.keycode==88 and ctrl and event.keysym.lower() != "x":
        event.widget.event_generate("<<Cut>>")

    if event.keycode==86 and ctrl and event.keysym.lower() != "v":
        event.widget.event_generate("<<Paste>>")
    
    if event.keycode==67 and ctrl and event.keysym.lower() != "c":
        event.widget.event_generate("<<Copy>>")

def send():
    msg = EntryBox.get('1.0', 'end-1c').strip()
    EntryBox.delete('0.0', END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        res = predict_class(msg)
        ChatLog.insert(END, 'Labedl is: ' + res + '\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title('AI App')
base.geometry('400x500')
base.resizable(width=True, height=True)

ChatLog = Text(base, bd=0, bg='white', height='8', width='50', font='Arial')
ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base , command=ChatLog.yview, cursor='heart')
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=('verdana', 12, 'bold'), text='Send', width='12',
                     height=5, bd=0, bg='#32de97', activebackground='#3c9d9b'
                    , fg='#ffffff', command= send)

EntryBox = Text(base, bd=0, bg='white', width='29', height='5', font='Arival')
EntryBox.bind_all('<Key>', _onKeyRelease, '+')
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6,y=6,height=386,width=370)
EntryBox.place(x=128,y=401,height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
