from flask import Flask, render_template, redirect

app = Flask(__name__)

@app.route("/")
def home():
    return redirect("http://tmwi.ddns.net/test2/test.php", code=302)
    
@app.route("/rator")
def rator():
    return render_template("rator.html")
    
if __name__ == "__main__":
    app.run(debug=False)
