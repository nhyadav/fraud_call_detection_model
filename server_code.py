from flask import Flask, request
from werkzeug.utils import secure_filename
import speech_recognition as sr
from googletrans import Translator
import joblib
from fraud_model_tfidf import cv
from pydub import AudioSegment

app = Flask(__name__)

"""changinfile extenstion"""

""""function voice data extraction"""
def getextractdatafromaudio(audiodata):
    language='english'
    p = Translator(service_urls=['translate.googleapis.com', 'translate.google.com', 'translate.google.co.kr'])
    r = sr.Recognizer()
    mic = sr.AudioFile(audiodata)
    with mic as source :
        r.adjust_for_ambient_noise(source)
        audio = r.record(source)
    try:
        result = r.recognize_google(audio)
        if language == 'hindi':
            k = p.translate(result, dest='en', src='auto', )
            translated = str(k.text)
            return translated
        else:
            return result

    except sr.UnknownValueError as e :
        return e
    except Exception as es:
        return es

def frauddetect(data):
    li_data =[data]
    # result = [str(wrd)]
    model = joblib.load('fraud_detection_model.sav')
    result_dta = cv.transform(li_data)
    ans = model.predict(result_dta)
    if ans[0] == 1 :
        return "normal"
    else :
        return "fraud"

@app.route('/api/upload/', methods=['GET','POST'])
def uploading():
    if request.method == 'POST':
        filedata = request.files['data']
        # flname = filedata.save(secure_filename(filedata.filename))
        result = getextractdatafromaudio(filedata)
        final_result = frauddetect(result)
    return final_result;



app.run(host='0.0.0.0', debug=True)
