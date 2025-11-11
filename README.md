# AI Highway Route Planner 🚀

A professional Streamlit dashboard that analyzes satellite imagery to generate optimal highway routes using land cover segmentation, terrain cost modeling, and A* pathfinding.

## 🌐 Live Demo
**Try it now:** [https://ai-route-optimizer.onrender.com](https://ai-route-optimizer.onrender.com)

> **Note:** The app may take 30-60 seconds to wake up on first visit (free tier limitation)

## ✨ Features
- 📤 Upload satellite imagery
- 🔲 Land cover segmentation (rule-based placeholder)
- 💰 Terrain cost map generation with adjustable costs
- 🗺️ A* pathfinding across cost-weighted grid
- 🛣️ Route visualization over satellite + cost map
- 📊 Detailed statistics (cost, length, efficiency)
- 💾 Export individual artifacts + ZIP bundle
- 🎨 Modern SaaS-style dark dashboard UI

## 🏗 Tech Stack
- **Python**
- **Streamlit** for UI
- **NumPy / OpenCV** for image & route rendering
- **Matplotlib** for cost map visualization

## 📦 Installation
```powershell
# Clone the repository
git clone https://github.com/ka-amith007/Route-optimatization-.git
cd Route-optimatization-

# (Recommended) Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Run the App
```powershell
streamlit run app.py
```
The app will open at: `http://localhost:8501` (Streamlit will display the exact port).

## 🖥 Workflow Stages
| Step | Name | Description |
|------|------|-------------|
| 1 | Upload | Provide satellite image input |
| 2 | Segmentation | Classify terrain (rule-based prototype) |
| 3 | Cost Map | Assign construction difficulty values |
| 4 | Route Planning | Compute optimal path with A* |
| 5 | Results | Review visuals & download assets |

## 📊 Land Cover Classes
| ID | Label   | Color      | Default Cost |
|----|---------|------------|--------------|
| 0  | Water   | Blue       | 1000 |
| 1  | Forest  | Green      | 500 |
| 2  | Urban   | Gray       | 200 |
| 3  | Barren  | Tan        | 100 |
| 4  | Road    | Black      | 50  |

## 📈 Example Enhancements (Future)
- ✅ Replace rule-based segmentation with trained U-Net / DeepLabV3+
- ✅ Interactive point selection via image click
- ✅ Caching intermediate computations
- ✅ Multi-route comparison (cost vs length vs impact)
- ✅ Environmental impact scoring

## 🗂 Project Structure
```
app.py                # Streamlit UI
src/
  segmentation.py     # Land cover segmentation
  cost_map.py         # Terrain cost mapping
  pathfinding.py      # A* route planning
  visualization.py    # Rendering helpers
  image_collection.py # Image IO
results/              # Generated artifacts (saved examples)
requirements.txt      # Dependencies
README.md             # Project documentation
```

## 🧪 Testing Idea (Not Implemented Yet)
Add unit tests for:
- A* path validity
- Cost map generation correctness
- Segmentation class balance

## 🐛 Known Limitations
- Segmentation is a color-threshold placeholder
- No geographic projection support
- Start/end points are numeric inputs (no map clicks yet)

## 🙌 Contributing
PRs and issues welcome! Ideas for enhancement are listed above.

## 📄 License
Consider adding a license (e.g., MIT) if you intend open collaboration.

---
**Made with 💡 AI and Streamlit**
