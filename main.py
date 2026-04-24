import requests
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

app = FastAPI()

conn = sqlite3.connect("sneaker_platform.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS sneakers(
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT,
brand TEXT,
retail REAL,
market REAL,
volume REAL,
year INTEGER,
image TEXT
)
""")

def collect_data():
    url = "https://api.sneaks-api.com/v2/sneakers?limit=100"
    r = requests.get(url)
    data = r.json()
    for s in data["results"]:
        if "retailPrice" not in s or s["retailPrice"] is None:
            continue
        retail = s["retailPrice"]
        market = retail * np.random.uniform(1.1,1.6)
        volume = np.random.randint(100,5000)
        year = np.random.randint(2015,2024)
        cursor.execute(
            "INSERT INTO sneakers(name,brand,retail,market,volume,year,image) VALUES(?,?,?,?,?,?,?)",
            (
                s.get("name", ""),
                s.get("brand", ""),
                retail,
                market,
                volume,
                year,
                s.get("image", {}).get("original", "")
            )
        )
    conn.commit()

def load_df():
    df = pd.read_sql("SELECT * FROM sneakers",conn)
    return df

scaler = StandardScaler()
price_model = RandomForestRegressor(n_estimators=300)

def train_price_model():
    df = load_df()
    if len(df) < 10:
        return
    X = df[["retail","volume","year"]]
    y = df["market"]
    X_scaled = scaler.fit_transform(X)
    price_model.fit(X_scaled,y)

def predict_price(retail,volume,year):
    X = np.array([[retail,volume,year]])
    X_scaled = scaler.transform(X)
    p = price_model.predict(X_scaled)
    return float(p[0])

class HypeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,1)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

hype_model = HypeNet()
def hype_score(retail,volume,year):
    x = torch.tensor([retail,volume,year],dtype=torch.float32)
    score = hype_model(x).item()
    return round(score*100,2)

@app.get("/update")
def update():
    collect_data()
    train_price_model()
    return {"status":"database updated and AI trained"}

@app.get("/api/sneakers")
def sneakers():
    rows = cursor.execute("SELECT * FROM sneakers LIMIT 500").fetchall()
    data = []
    for r in rows:
        prediction = predict_price(r[3],r[5],r[6])
        hype = hype_score(r[3],r[5],r[6])
        data.append({
            "id": r[0],
            "name":r[1],
            "brand":r[2],
            "retail":r[3],
            "market":r[4],
            "prediction":prediction,
            "hype":hype,
            "year":r[6],
            "volume": r[5],
            "image":r[7]
        })
    return data

@app.get("/", response_class=HTMLResponse)
def homepage():
    html = """
<!DOCTYPE html>
<html>
<head>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Sneaker Wraith</title>
<style>
body {
  font-family: Arial, sans-serif;
  background: #fff;
  color: #000;
  margin: 0;
}
header {
  background: #fff;
  color: #000;
  padding: 20px 20px 10px 20px;
  border-bottom: 1px solid #eee;
}
h1 {
  font-size: 2.1rem;
  margin: 0 0 10px 0;
  font-weight: bold;
}
#grid {
  display: grid;
  grid-template-columns: repeat(auto-fill,minmax(180px,1fr));
  gap: 20px;
  padding: 20px;
}
.card {
  background: #fff;
  color: #000;
  border-radius: 10px;
  border: 1px solid #eee;
  cursor: pointer;
  transition: transform 0.1s;
}
.card:hover {
  transform: translateY(-3px) scale(1.025);
  box-shadow: 0 6px 25px 0 rgba(0,0,0,0.08);
}
.card img {
  width: 100%;
  display: block;
  border-radius: 10px 10px 0 0;
  object-fit: contain;
  aspect-ratio: 1/1;
  background: #fafafa;
}
.card .card-name {
  text-align: center;
  font-weight: 600;
  font-size: 1.07rem;
  padding: 11px 6px 13px 6px;
  margin: 0;
  color: #222;
}
#modal-bg {
  position: fixed;
  top:0;left:0;right:0;bottom:0;
  background: rgba(0,0,0,0.38);
  display: none;
  justify-content: center;
  align-items: center;
  z-index: 101;
}
#modal {
  background: #fff;
  color: #000;
  border-radius: 14px;
  min-width: 320px;
  max-width: 95vw;
  box-shadow: 0 10px 34px 0 rgba(0,0,0,0.20);
  padding: 30px;
  text-align: left;
  position: relative;
}
#modal img {
  width: 85%;
  margin: 0 0 20px 0;
  display: block;
  border-radius: 10px;
  box-shadow: 0 3px 13px 0 rgba(0,0,0,0.06);
  object-fit: contain;
  margin-left: auto; margin-right: auto;
  background: #fafafa;
}
.close-btn {
  position: absolute;
  right: 10px; top: 10px;
  background: #000;
  color: #fff; border: none;
  font-size: 1.2rem; padding: 0 12px 0 12px;
  border-radius: 15px; cursor: pointer;
  height: 32px; width: 32px;
  display: flex; align-items: center; justify-content: center;
}
.close-btn:hover{background: #222}
.details-table {
  margin-top: 14px; width: 100%; font-size: 1.06rem;
  border-collapse: collapse;
}
.details-table td {
  padding: 6px 11px 6px 0; vertical-align: top;
  border: none;
}
@media (max-width: 600px) {
  #modal { padding: 15px; }
}
</style>
</head>
<body>
<header>
  <h1>Sneaker Wraith</h1>
</header>
<div id=\"grid\"></div>

<div id=\"modal-bg\">
  <div id=\"modal\">
    <button class=\"close-btn\" onclick=\"closeModal()\">×</button>
    <img id=\"modal-img\" src=\"\" alt=\"Sneaker photo\">
    <div id=\"modal-desc\">
      <!-- Details added with JS -->
    </div>
  </div>
</div>

<script>
let sneakerData = []
async function load(){
  const res = await fetch("/api/sneakers")
  sneakerData = await res.json()
  const grid = document.getElementById("grid")
  grid.innerHTML=""
  sneakerData.forEach(s=>{
    const safeName = s.name ? s.name.replace(/\"/g,"&quot;") : '';
    const safeImage = s.image || "";
    grid.innerHTML += `
    <div class="card" onclick="showModal(${s.id})">
      <img src="${safeImage}" alt="${safeName}">
      <div class="card-name">${safeName}</div>
    </div>
    `
  })
}
function showModal(id){
  const s = sneakerData.find(x=>x.id===id);
  if(!s) return;
  document.getElementById("modal-img").src = s.image || "";
  document.getElementById("modal-desc").innerHTML = `
    <table class='details-table'>
      <tr><td style="font-weight:600">Name:</td><td>${s.name || ''}</td></tr>
      <tr><td style="font-weight:600">Brand:</td><td>${s.brand || ''}</td></tr>
      <tr><td style="font-weight:600">Retail Price:</td><td>$${s.retail}</td></tr>
      <tr><td style="font-weight:600">Year:</td><td>${s.year}</td></tr>
      <tr><td style="font-weight:600">Volume:</td><td>${s.volume}</td></tr>
      <tr><td style="font-weight:600">Market Price:</td><td>$${s.market}</td></tr>
      <tr><td style="font-weight:600">AI Prediction:</td><td>$${s.prediction}</td></tr>
      <tr><td style="font-weight:600">Hype Score:</td><td>${s.hype}</td></tr>
    </table>
  `;
  document.getElementById("modal-bg").style.display="flex";
}
function closeModal(){
  document.getElementById("modal-bg").style.display="none";
}
window.onclick = function(event){
  if(event.target===document.getElementById("modal-bg")){
    closeModal();
  }
}
load()
</script>

</body>
</html>
"""
    return html
