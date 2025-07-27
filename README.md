# 🎵 Beat Addicts - AI Music Generation Studio

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://github.com/djtlb/NewBeats)

**Professional AI-powered music generation studio with unified workflow and advanced optimization**

## ✨ Latest Major Update - Complete Studio Workflow

🎼 **Unified Song Creation** - Create complete songs with AI lyrics, music, and vocals from a single button!
🤖 **Enhanced AI Lyrics** - 7 genre-specific templates with professional formatting
⚡ **AMD GPU Support** - Optimized for AMD Radeon GPUs with CPU acceleration
🎯 **Smart Workflow** - Seamless lyrics → music → vocals integration

---

## 🚀 Key Features

### 🎵 **Complete Music Creation Pipeline**

- **One-Click Song Generation** - Unified workflow from concept to complete song
- **AI Lyrics Generator** - Genre-specific templates for Hip Hop, Pop, Rock, Country, R&B, Electronic, Reggae
- **Professional Music Generation** - High-quality AI music with 3 quality levels
- **AI Vocals Synthesis** - 5 voice styles with natural-sounding speech
- **Real-time Progress Tracking** - Visual progress through entire creation process

### 🎨 **Intelligent Lyrics Options**

- **Auto-Generate** - AI creates lyrics based on music genre
- **Theme-Based** - Generate lyrics from custom themes
- **Custom Lyrics** - Write your own lyrics
- **Instrumental** - Pure music without vocals

### ⚡ **Performance & Optimization**

- **AMD GPU Detection** - Automatic hardware detection and optimization
- **Multi-Core CPU Boost** - 8-thread optimization with MKL-DNN acceleration
- **Memory Efficient** - Optimized algorithms for CPU generation
- **Professional Audio Processing** - Enhanced post-processing pipeline

### 🎛️ **Professional Controls**

- **Quality Levels**: Standard, High, Professional
- **Duration Options**: 10-60 seconds with optimized generation
- **Voice Styles**: Default, Female, Male, Robot, Narrator
- **Genre Intelligence**: Context-aware generation based on musical style

---

## 🛠️ Installation & Setup

### Requirements

- **Python 3.11+**
- **PyTorch 2.5.1+**
- **8GB+ RAM** (16GB recommended)
- **AMD/NVIDIA GPU** (optional, CPU optimized)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/djtlb/NewBeats.git
cd NewBeats

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Access the Studio

Open your browser to `http://localhost:5000` and start creating music!

---

## 🎯 How to Use

### 1. **Describe Your Music**

Enter the genre, style, instruments, and mood you want:

```
"Upbeat electronic dance music with heavy bass and synthesizers"
```

### 2. **Choose Lyrics Option**

- **Auto-Generate**: AI creates lyrics from your music style
- **Theme-Based**: Provide a theme like "love and heartbreak"
- **Custom**: Write your own lyrics
- **Instrumental**: Music only

### 3. **Set Generation Parameters**

- **Duration**: 10-60 seconds
- **Quality**: Standard/High/Professional
- **Vocals**: Optional AI-generated vocals

### 4. **Create Complete Song**

Click "🎵 Create Complete Song" and watch the magic happen!

---

## 🔧 Technical Architecture

### **AI Models & Processing**

- **MusicGen**: Meta's state-of-the-art music generation
- **GPT-2**: Enhanced with genre-specific templates for lyrics
- **Professional Audio Pipeline**: Multi-stage post-processing
- **Optimized Inference**: CPU/GPU hybrid processing

### **Performance Optimizations**

- **Multi-threaded Processing**: 8-core CPU utilization
- **MKL-DNN Acceleration**: Enhanced neural network performance
- **Memory Management**: Efficient model loading and caching
- **Hardware Detection**: Automatic AMD/NVIDIA GPU optimization

### **Web Interface**

- **Flask Backend**: RESTful API with real-time progress
- **Responsive Frontend**: Modern HTML5/CSS3/JavaScript
- **Progressive Enhancement**: Works on all devices
- **Real-time Updates**: WebSocket-like progress tracking

---

## 📊 System Requirements & Performance

### **Hardware Support**

- **CPU**: Optimized for multi-core processors (8+ cores recommended)
- **AMD GPU**: Radeon RX 5700+ (with ROCm on Linux)
- **NVIDIA GPU**: GTX 1660+ / RTX series (CUDA support)
- **Apple Silicon**: M1/M2 with Metal Performance Shaders

### **Performance Metrics**

- **Generation Speed**: 15-60 seconds per song
- **Quality Options**: 3 levels (Standard/High/Professional)
- **Concurrent Users**: Supports multiple sessions
- **Memory Usage**: 2-6GB depending on quality

---

## 🎼 Advanced Features

### **Genre-Specific Intelligence**

Each genre has optimized templates and processing:

- **Hip Hop**: Strong rhythmic patterns, rap-style lyrics
- **Pop**: Catchy melodies, commercial structure
- **Rock**: Guitar-driven arrangements, powerful vocals
- **Electronic**: Synthesizer focus, electronic elements
- **Country**: Traditional instruments, storytelling lyrics
- **R&B**: Smooth vocals, rhythm-focused
- **Reggae**: Characteristic rhythm patterns, cultural elements

### **Professional Workflows**

- **A&R Mode**: Generate multiple variations quickly
- **Demo Creation**: Fast prototyping for songwriters
- **Commercial Production**: High-quality output for releases
- **Educational Use**: Learning tool for music composition

---

## 🔍 API Reference

### **Generate Complete Song**

```python
POST /api/generate
{
    "genre": "Electronic dance music",
    "lyrics": "Auto-generated or custom",
    "duration": 30,
    "quality": "high",
    "include_vocals": true,
    "voice_style": "female"
}
```

### **Generate AI Lyrics**

```python
POST /api/generate-lyrics
{
    "theme": "love and adventure",
    "genre": "pop"
}
```

### **Track Progress**

```python
GET /api/progress/{session_id}
```

---

## 🤝 Contributing

We welcome contributions! Areas where you can help:

- **New AI Models**: Integration of latest music generation models
- **Genre Expansion**: Additional genre templates and styles
- **Performance**: CPU/GPU optimization improvements
- **UI/UX**: Interface enhancements and user experience
- **Documentation**: Tutorials, examples, and guides

---

## 📜 License & Credits

### **License**

MIT License - See [LICENSE](LICENSE) for details

### **Credits**

- **Meta AI**: MusicGen model and AudioCraft framework
- **OpenAI**: GPT-2 for lyrics generation
- **Hugging Face**: Model hosting and transformers library
- **PyTorch**: Deep learning framework
- **Flask**: Web framework

---

## 📈 Roadmap

### **Upcoming Features**

- 🎸 **Instrument Separation**: Individual track control
- 🎭 **Style Transfer**: Convert songs between genres
- 🎤 **Voice Cloning**: Custom voice models
- 🎵 **MIDI Export**: Sheet music and MIDI generation
- 🔊 **Master Quality**: Professional mastering pipeline
- 🌐 **Cloud Processing**: GPU acceleration in the cloud

---

**🎵 Start creating professional AI music today with Beat Addicts!**

*Made with ❤️ by the AI music community*
