# 🎯 Quick Start Guide - Local Testing

## Run the Streamlit App Locally

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Run the Streamlit App**
```bash
streamlit run app.py
```

### 3. **Access the App**
- Automatically opens at: `http://localhost:8501`
- If not, manually go to that URL in your browser

---

## 📱 App Features

### **🏠 Home Page**
- Project overview
- Key features summary
- Quick navigation guide

### **📊 Dataset Explorer**
- Select any of 5 domains
- View dataset statistics
- See class distribution
- Preview dataset samples

### **🤖 Model Training**
- Select domain to train on
- Adjust learning rate (0.001 - 0.1)
- Adjust epochs (50 - 300)
- Train two models:
  - Logistic Regression (Baseline)
  - Multi-Layer Perceptron (Advanced)
- Real-time metrics display
- Loss curves visualization

### **📈 Performance Analysis**
- Compare model metrics (Accuracy, Precision, Recall, F1-Score)
- Visual comparison charts
- Detailed metrics table
- Side-by-side bar charts

### **🔮 Make Predictions**
- Enter custom feature values
- Get predictions from both models
- See confidence scores
- Visualize prediction probabilities

---

## 🛠️ Troubleshooting

### App doesn't start?
```bash
# Clear Streamlit cache
streamlit cache clear

# Try again
streamlit run app.py
```

### Missing modules?
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Port already in use?
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

---

## 📝 Configuration

Edit `.streamlit/config.toml` to customize:
- **Theme colors** (primaryColor, backgroundColor, etc.)
- **Server settings** (max upload size, etc.)
- **Logger level** (debug, info, warning, error)

---

## ✅ Before Deploying to Streamlit Cloud

1. Test app locally: ✅ `streamlit run app.py`
2. Verify all features work
3. Check that training completes successfully
4. Test predictions feature
5. Push to GitHub when satisfied

---

**Happy testing! 🚀**
