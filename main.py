#main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import folium
from folium.plugins import HeatMap, MousePosition, Fullscreen
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from contextlib import asynccontextmanager
import asyncio
import time
from fastapi.middleware.cors import CORSMiddleware
from data_downloader.downloader import PeriodicMongoDBDataDownloader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Configuration using absolute paths
DATA_PATH = os.path.join(current_dir, 'data')
STATIC_PATH = os.path.join(current_dir, 'static')
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(STATIC_PATH, exist_ok=True)

# Initialize downloader
downloader = PeriodicMongoDBDataDownloader(output_dir=DATA_PATH)

def load_and_preprocess_data():
    """Load and preprocess data with enhanced validation checks"""
    try:
        # Load datasets with error handling
        crime_records = pd.read_csv(
            os.path.join(DATA_PATH, 'crime_records.csv'),
            sep=',',
            dtype={'crime_type': 'string', 'location': 'string', 'date': 'string', 'time': 'string'}
        )
        
        # Create datetime column with proper error handling
        crime_records['datetime'] = pd.to_datetime(
            crime_records['date'] + ' ' + crime_records['time'],
            errors='coerce'
        )
        
        locations = pd.read_csv(
            os.path.join(DATA_PATH, 'locations.csv'),
            sep=',',
            dtype={'_id': 'string'}
        )
        crime_types = pd.read_csv(
            os.path.join(DATA_PATH, 'crime_types.csv'),
            sep=',',
            dtype={'_id': 'string'}
        )

        # Clean column names
        crime_records.columns = crime_records.columns.str.strip().str.lower()
        locations.columns = locations.columns.str.strip().str.lower()
        crime_types.columns = crime_types.columns.str.strip().str.lower()

        # Merge datasets with validation
        merged_df = (
            crime_records
            .merge(locations, left_on='location', right_on='_id', how='inner')
            .merge(crime_types, left_on='crime_type', right_on='_id', how='inner')
        )

        # Handle merged column names
        merged_df = merged_df.rename(columns={
            'crime_type_y': 'crime_type_name',
            'case_status': 'status'
        })

        # Validate coordinates
        merged_df['latitude'] = pd.to_numeric(merged_df['latitude'], errors='coerce')
        merged_df['longitude'] = pd.to_numeric(merged_df['longitude'], errors='coerce')
        coord_mask = (
            merged_df['latitude'].between(-90, 90) & 
            merged_df['longitude'].between(-180, 180) & 
            merged_df['latitude'].notna() & 
            merged_df['longitude'].notna()
        )
        merged_df = merged_df[coord_mask].copy()

        if merged_df.empty:
            raise ValueError("No valid geographic data remaining after coordinate cleaning")

        # Validate datetime
        merged_df = merged_df.dropna(subset=['datetime'])
        if not pd.api.types.is_datetime64_any_dtype(merged_df['datetime']):
            raise ValueError("Datetime column is not properly formatted")

        if merged_df.empty:
            raise ValueError("No valid timestamps remaining after datetime cleaning")

        # Create temporal features
        merged_df['hour'] = merged_df['datetime'].dt.hour
        merged_df['day_of_week'] = merged_df['datetime'].dt.dayofweek
        merged_df['month'] = merged_df['datetime'].dt.month

        # Spatial clustering with validation
        coords = merged_df[['latitude', 'longitude']].values
        if len(coords) > 1:
            n_clusters = min(10, len(coords))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            merged_df['cluster'] = kmeans.fit_predict(coords)
        else:
            merged_df['cluster'] = 0

        return merged_df, kmeans

    except Exception as e:
        raise RuntimeError(f"Data processing failed: {str(e)}")

def generate_heatmap(df):
    """Generate heatmap visualization"""
    try:
        base_lat = df['latitude'].mean()
        base_lon = df['longitude'].mean()
        
        heatmap = folium.Map(location=[base_lat, base_lon], zoom_start=12)
        HeatMap(
            df[['latitude', 'longitude', 'risk_score']].values.tolist(),
            radius=15,
            blur=20,
            gradient={
                '0.5': '#0000ff',     
                '0.6': '#00ff00',
                '0.7': '#ffff00',     
                '0.8': '#ff3333', 
                '0.9': '#8b0000'      
            }
        ).add_to(heatmap)
        
        MousePosition().add_to(heatmap)
        Fullscreen().add_to(heatmap)
        heatmap.save(os.path.join(STATIC_PATH, 'heatmap.html'))
        return True
    except Exception as e:
        raise RuntimeError(f"Heatmap generation failed: {str(e)}")

def generate_hotspot_map(df, kmeans_model):
    """Generate hotspot map with cluster markers"""
    try:
        base_lat = df['latitude'].mean()
        base_lon = df['longitude'].mean()

        # Get top 5 hotspots
        cluster_risk = df.groupby('cluster')['risk_score'].mean().reset_index()
        top_clusters = cluster_risk.nlargest(5, 'risk_score')['cluster'].values
        
        hotspots = []
        for cluster in top_clusters:
            center = kmeans_model.cluster_centers_[cluster]
            risk = cluster_risk.loc[cluster_risk['cluster'] == cluster, 'risk_score'].values[0]
            count = df[df['cluster'] == cluster].shape[0]
            
            hotspots.append({
                "latitude": center[0],
                "longitude": center[1],
                "risk_score": risk,
                "crime_count": count
            })

        # Create hotspot map
        hotspot_map = folium.Map(location=[base_lat, base_lon], zoom_start=12)
        
        # Add markers for hotspots
        for hotspot in hotspots:
            folium.CircleMarker(
                location=[hotspot['latitude'], hotspot['longitude']],
                radius=10 + (hotspot['risk_score'] * 10),
                color='#ff4444',
                fill=True,
                fill_color='#ff0000',
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"""🚨 **High-Risk Hotspot** 🚨
                    🔥 **Risk Score**: {hotspot['risk_score']:.2f}<br>
                    📍 Coordinates: {hotspot['latitude']:.4f}, {hotspot['longitude']:.4f}<br>
                    ⚠️ **Crime Count**: {hotspot['crime_count']}<br>
                    <i>This area requires increased monitoring</i>""",
                    max_width=250,
                    max_height=150
                )
            ).add_to(hotspot_map)
        
        MousePosition().add_to(hotspot_map)
        Fullscreen().add_to(hotspot_map)
        hotspot_map.save(os.path.join(STATIC_PATH, 'hotspot_map.html'))
        return True
        
    except Exception as e:
        raise RuntimeError(f"Hotspot map generation failed: {str(e)}")
    
def generate_analysis_maps(df, kmeans_model):
    """Generate all map visualizations"""
    try:
        # Status Map
        status_map = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
        
        # Status color mapping
        status_colors = {
            'Closed': 'green',
            'Open': 'red',
            'Pending': 'blue',

            'Resolved': 'green',
            'Ongoing': 'red',
            'Under Investigation': 'blue'
        }
        
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                color=status_colors.get(row['status'].title(), 'blue'),
                fill=True,
                fill_color=status_colors.get(row['status'].title(), 'blue'),
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{row.get('crime_type_name', 'N/A')}</b><br>"
                    f"Status: {row.get('status', 'N/A')}<br>"
                    f"Date: {row.get('date', 'N/A')}",
                    max_width=300
                ),
                tooltip=f"Status: {row.get('status', 'N/A')}"
            ).add_to(status_map)
        status_map.save(os.path.join(STATIC_PATH, 'status_map.html'))

        # Risk Prediction Model
        features = ['hour', 'day_of_week', 'month', 'latitude', 'longitude', 'cluster']
        target = 'crime_occurred_indoors_or_outdoors'
        
        model = Pipeline([
            ('preprocessor', ColumnTransformer([('num', StandardScaler(), features)])),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        model.fit(df[features], df[target])
        df['risk_score'] = model.predict_proba(df[features])[:, 1]
        df['risk_score'] = pd.to_numeric(df['risk_score'], errors='coerce').fillna(0)

        # Generate separate maps
        generate_heatmap(df)
        generate_hotspot_map(df, kmeans_model)
        return True

    except Exception as e:
        raise RuntimeError(f"Map generation failed: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management with periodic updates"""
    try:
        print("🚀 Initializing application...")
        
        # Initial data processing
        downloader.start_single_download(['crime_records', 'crime_types', 'locations'])
        df, kmeans_model = load_and_preprocess_data()
        
        if df.empty:
            raise ValueError("No valid data available")
            
        print(f"✅ Processed {len(df)} records")
        generate_analysis_maps(df, kmeans_model)
        app.state.initialized = True
        print("🗺️ Maps generated successfully")
        
        # Start periodic updates
        async def periodic_update():
            while True:
                await asyncio.sleep(1800)  # Wait for 30 minutes (1800 seconds)
                try:
                    print("🔄 Starting periodic update...")
                    downloader.start_single_download(['crime_records', 'crime_types', 'locations'])
                    new_df, new_kmeans = load_and_preprocess_data()
                    generate_analysis_maps(new_df, new_kmeans)
                    app.state.initialized = True
                    print("✅ Periodic update completed successfully")
                except Exception as e:
                    print(f"❌ Periodic update failed: {str(e)}")
        
        asyncio.create_task(periodic_update())

    except Exception as e:
        app.state.initialized = False
        print(f"❌ Initialization failed: {str(e)}")
        raise
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOWED_ORIGINS")],  # Get allowed origin from environment variable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not app.state.initialized:
        return HTMLResponse("<h1>Initializing... Please refresh shortly</h1>", status_code=503)
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/heat-map", response_class=HTMLResponse)
async def get_heat_map():
    if not app.state.initialized:
        raise HTTPException(503, detail="Service initializing")
    return FileResponse(os.path.join(STATIC_PATH, 'heatmap.html'))

@app.get("/hotspot-map", response_class=HTMLResponse)
async def get_hotspot_map():
    if not app.state.initialized:
        raise HTTPException(503, detail="Service initializing")
    return FileResponse(os.path.join(STATIC_PATH, 'hotspot_map.html'))

@app.get("/status-map", response_class=HTMLResponse)
async def get_status_map():
    if not app.state.initialized:
        raise HTTPException(503, detail="Service initializing")
    return FileResponse(os.path.join(STATIC_PATH, 'status_map.html'))

@app.get("/hotspot-data", response_class=JSONResponse)
async def get_hotspot_data():
    if not app.state.initialized:
        raise HTTPException(503, detail="Service initializing")
    
    df, kmeans_model = load_and_preprocess_data()
    features = ['hour', 'day_of_week', 'month', 'latitude', 'longitude', 'cluster']
    target = 'crime_occurred_indoors_or_outdoors'
    
    model = Pipeline([
        ('preprocessor', ColumnTransformer([('num', StandardScaler(), features)])),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(df[features], df[target])
    df['risk_score'] = model.predict_proba(df[features])[:, 1]
    
    cluster_risk = df.groupby('cluster')['risk_score'].mean().reset_index()
    top_clusters = cluster_risk.nlargest(5, 'risk_score')['cluster'].values
    
    return {
        "hotspots": [
            {
                "latitude": kmeans_model.cluster_centers_[cluster][0],
                "longitude": kmeans_model.cluster_centers_[cluster][1],
                "risk_score": cluster_risk.loc[cluster_risk['cluster'] == cluster, 'risk_score'].values[0],
                "crime_count": df[df['cluster'] == cluster].shape[0]
            }
            for cluster in top_clusters
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "OK" if app.state.initialized else "Initializing"}

# API endpoints to expose HTML maps
@app.get("/api/heatmap", response_class=HTMLResponse)
async def get_heatmap_api():
    if not app.state.initialized:
        raise HTTPException(503, detail="Service initializing")
    return FileResponse(os.path.join(STATIC_PATH, 'heatmap.html'))

@app.get("/api/hotspot-map", response_class=HTMLResponse)
async def get_hotspot_map_api():
    if not app.state.initialized:
        raise HTTPException(503, detail="Service initializing")
    return FileResponse(os.path.join(STATIC_PATH, 'hotspot_map.html'))

@app.get("/api/status-map", response_class=HTMLResponse)
async def get_status_map_api():
    if not app.state.initialized:
        raise HTTPException(503, detail="Service initializing")
    return FileResponse(os.path.join(STATIC_PATH, 'status_map.html'))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)