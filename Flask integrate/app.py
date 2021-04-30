from flask import Flask,url_for,render_template,request
import numpy as np
import pickle
model=pickle.load(open('model.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check',methods=['POST'])
def check():
    val=[int(x) for x in request.form.values()]
    fnl=[np.array(val)]
    result=model.predict(fnl)
    return str(result)


if __name__=="__main__":
    app.run(debug=True)



