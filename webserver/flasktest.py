from flask import Flask, render_template      

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")
    
@app.route("/rator")
def rator():
    return render_template("rator.html")
    
if __name__ == "__main__":
    app.run(debug=True)
