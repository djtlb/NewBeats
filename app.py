from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import threading
from pathlib import Path
import json
from datetime import datetime

# Import your beat-addicts modules
from beat_addicts.melody_generator import MelodyGenerator
from beat_addicts.song_builder import SongBuilder
from beat_addicts.lyrics_generator import LyricsGenerator

app = Flask(__name__)
CORS(app)

# Configuration
OUTPUT_DIR = Path("web_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Global variables for progress tracking
generation_progress = {}
active_generators = {}

class WebMelodyGenerator:
    """Web-optimized wrapper for MelodyGenerator with progress tracking."""
    
    def __init__(self, model_size="small"):
        self.generator = MelodyGenerator(model_size=model_size)
        self.lyrics_gen = LyricsGenerator()
        
    def get_device_info(self):
        return self.generator.get_device_info()
        
    def generate_song(self, genre_prompt, lyrics_prompt, duration, session_id):
        """Generate a complete song with progress tracking."""
        try:
            # Initialize progress
            generation_progress[session_id] = {
                'status': 'starting',
                'progress': 0,
                'stage': 'Initializing generation...',
                'error': None
            }
            
            def update_progress(progress, stage="Generating..."):
                generation_progress[session_id].update({
                    'progress': int(progress * 100),
                    'stage': stage
                })
            
            # Stage 1: Generate lyrics (if custom lyrics not provided)
            update_progress(0.1, "Generating lyrics...")
            if lyrics_prompt.strip():
                lyrics = lyrics_prompt
            else:
                lyrics = self.lyrics_gen.generate_lyrics(genre_prompt)
            
            # Stage 2: Build song description
            update_progress(0.3, "Building music prompt...")
            
            # Create a more effective prompt for MusicGen
            if lyrics_prompt.strip():
                # If user provided lyrics, use them to enhance the genre
                music_prompt = f"{genre_prompt} with {lyrics_prompt.strip()}"
            else:
                # If no lyrics, use just the genre with some musical descriptors
                music_prompt = f"{genre_prompt} music"
            
            print(f"🎯 Using music prompt: '{music_prompt}'")
            
            # Stage 3: Generate melody
            update_progress(0.4, "Generating melody...")
            
            def progress_callback(progress):
                # Map melody generation progress to 40-90% of total
                total_progress = 0.4 + (progress * 0.5)
                update_progress(total_progress, "Generating melody...")
            
            melody = self.generator.generate(
                music_prompt, 
                duration=duration,
                progress_callback=progress_callback
            )
            
            # Stage 4: Save files
            update_progress(0.9, "Saving files...")
            
            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"song_{timestamp}_{session_id[:8]}"
            
            # Save audio
            audio_path = OUTPUT_DIR / f"{filename}.wav"
            self.generator.save(melody, str(audio_path.with_suffix('')), format="wav")
            
            # Save lyrics
            lyrics_path = OUTPUT_DIR / f"{filename}_lyrics.txt"
            with open(lyrics_path, 'w', encoding='utf-8') as f:
                f.write(f"Genre: {genre_prompt}\n\n")
                f.write(f"Generated Lyrics:\n{lyrics}\n\n")
                f.write(f"Music Prompt: {music_prompt}\n")
                f.write(f"Duration: {duration}s\n")
                f.write(f"Device: {self.generator.device}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
            
            # Complete
            update_progress(1.0, "Generation complete!")
            generation_progress[session_id].update({
                'status': 'complete',
                'audio_file': f"{filename}.wav",
                'lyrics_file': f"{filename}_lyrics.txt",
                'lyrics_content': lyrics
            })
            
            return {
                'success': True,
                'audio_file': f"{filename}.wav",
                'lyrics_file': f"{filename}_lyrics.txt",
                'lyrics_content': lyrics
            }
            
        except Exception as e:
            generation_progress[session_id].update({
                'status': 'error',
                'error': str(e)
            })
            return {'success': False, 'error': str(e)}

# Initialize generator
web_generator = WebMelodyGenerator()

@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')

@app.route('/api/device-info')
def device_info():
    """Get device information."""
    return jsonify(web_generator.get_device_info())

@app.route('/api/generate', methods=['POST'])
def generate_music():
    """Start music generation."""
    data = request.json
    
    # Validate input
    genre_prompt = data.get('genre', '').strip()
    lyrics_prompt = data.get('lyrics', '').strip()
    duration = min(int(data.get('duration', 15)), 60)  # Max 60 seconds for web
    
    if not genre_prompt:
        return jsonify({'error': 'Genre prompt is required'}), 400
    
    # Create session ID
    session_id = str(uuid.uuid4())
    
    # Start generation in background thread
    def generate():
        web_generator.generate_song(genre_prompt, lyrics_prompt, duration, session_id)
    
    thread = threading.Thread(target=generate)
    thread.daemon = True
    thread.start()
    
    return jsonify({'session_id': session_id})

@app.route('/api/progress/<session_id>')
def get_progress(session_id):
    """Get generation progress."""
    progress = generation_progress.get(session_id, {
        'status': 'not_found',
        'error': 'Session not found'
    })
    return jsonify(progress)

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download generated files."""
    file_path = OUTPUT_DIR / filename
    if file_path.exists():
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/examples')
def get_examples():
    """Get example prompts for the UI."""
    examples = {
        'genres': [
            "upbeat electronic dance music",
            "chill lo-fi hip hop",
            "jazz piano with drums",
            "rock with electric guitar",
            "ambient electronic",
            "classical piano",
            "reggae with bass",
            "country acoustic guitar",
            "trap beat with 808s",
            "house music with synths"
        ],
        'lyrics_themes': [
            "A song about chasing dreams and never giving up",
            "Love ballad about finding your soulmate",
            "Party anthem about having fun with friends",
            "Motivational song about overcoming challenges",
            "Nostalgic song about childhood memories",
            "Road trip adventure song",
            "Song about the beauty of nature",
            "Celebration of music and creativity"
        ]
    }
    return jsonify(examples)

@app.route('/api/history')
def get_history():
    """Get list of generated songs."""
    try:
        files = list(OUTPUT_DIR.glob("*.wav"))
        history = []
        
        for audio_file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
            lyrics_file = audio_file.with_name(audio_file.stem + "_lyrics.txt")
            
            item = {
                'audio_file': audio_file.name,
                'lyrics_file': lyrics_file.name if lyrics_file.exists() else None,
                'created': datetime.fromtimestamp(audio_file.stat().st_mtime).isoformat(),
                'size_mb': round(audio_file.stat().st_size / 1024 / 1024, 2)
            }
            
            # Try to read basic info from lyrics file
            if lyrics_file.exists():
                try:
                    with open(lyrics_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        for line in lines[:5]:  # Check first 5 lines
                            if line.startswith('Genre:'):
                                item['genre'] = line.replace('Genre:', '').strip()
                            elif line.startswith('Duration:'):
                                item['duration'] = line.replace('Duration:', '').strip()
                except:
                    pass
            
            history.append(item)
        
        return jsonify(history[:20])  # Return last 20 files
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎵 Starting Beat-addicts Web Interface...")
    print(f"📊 Device Info: {web_generator.get_device_info()}")
    print(f"📁 Output Directory: {OUTPUT_DIR.absolute()}")
    print("🌐 Access the web interface at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
