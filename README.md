📊 Features
🔥 Heatmap of high-risk areas based on predicted crime probability

📍 Hotspot detection using KMeans clustering

🗺️ Status map showing resolved, ongoing, or pending cases

🤖 Predictive risk scoring using Random Forest

🔁 Periodic automatic data updates from MongoDB

🧭 Fully interactive maps using Folium & Leaflet

🧪 API endpoints to fetch map visualizations and hotspot data


⚙️ Technologies Used
Backend: FastAPI, MongoDB

Frontend Maps: Folium

ML/Analysis: scikit-learn, pandas, KMeans, RandomForestClassifier

Dev Tools: uvicorn, python-dotenv, Jinja2, CORS, asyncio


📦 Setup Instructions
1. Clone the repo
git clone https://github.com/stannnzoneeee/Crime-Visualization.git
cd Crime-Visualization

2. Create .env file
MONGO_URI="YOUR MONGO DB URL"
MONGO_DB_NAME=MONGODB NAME
ALLOWED_ORIGINS=http://localhost:8000

3. Install dependencies
pip install -r requirements.txt

4. Run the app
uvicorn main:app --reload
