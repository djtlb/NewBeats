from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import threading
from pathlib import Path
import json
from datetime import datetime
import requests
import base64

# Import your beat-addicts modules
from beat_addicts.melody_generator import MelodyGenerator
from beat_addicts.song_builder import SongBuilder
from beat_addicts.lyrics_generator import LyricsGenerator

app = Flask(__name__)
CORS(app)

# Configuration
OUTPUT_DIR = Path("web_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# API Configuration for vocals/sounds generation
VOCALS_API_KEY = "sk_1ed36e032a4dc2e4fdfd92ee27be8ca6ff09df6edf714474"
VOCALS_API_BASE_URL = "https://api.elevenlabs.io/v1"  # Common API for voice generation

# Global variables for progress tracking
generation_progress = {}
active_generators = {}

class VocalsGenerator:
    """Vocals and sound effects generator using external API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = VOCALS_API_BASE_URL
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
    def generate_vocals(self, text: str, voice_style: str = "default", progress_callback=None) -> bytes:
        """Generate vocals from text using the API."""
        try:
            if progress_callback:
                progress_callback(0.1)
                
            # Voice mapping for different styles
            voice_models = {
                "default": "pNInz6obpgDQGcFmaJgB",  # Example voice ID
                "female": "EXAVITQu4vr4xnSDxMaL",
                "male": "VR6AewLTigWG4xSOukaG", 
                "robot": "onwK4e9ZLuTAKqWW03F9",
                "narrator": "pqHfZKP75CvOlQylNhV4"
            }
            
            voice_id = voice_models.get(voice_style, voice_models["default"])
            
            # Prepare request data
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            if progress_callback:
                progress_callback(0.5)
                
            # Make API request
            response = requests.post(url, json=data, headers=self.headers)
            
            if response.status_code == 200:
                if progress_callback:
                    progress_callback(0.9)
                return response.content
            else:
                raise Exception(f"API request failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Vocals generation failed: {e}")
            raise
            
    def generate_sound_effect(self, description: str, duration: int = 5, progress_callback=None) -> bytes:
        """Generate sound effects from description."""
        try:
            if progress_callback:
                progress_callback(0.1)
                
            # Sound effects endpoint (if available)
            url = f"{self.base_url}/sound-generation"
            data = {
                "text": description,
                "duration_seconds": duration,
                "prompt_influence": 0.3
            }
            
            if progress_callback:
                progress_callback(0.5)
                
            response = requests.post(url, json=data, headers=self.headers)
            
            if response.status_code == 200:
                if progress_callback:
                    progress_callback(0.9)
                return response.content
            else:
                # Fallback to text-to-speech for sound descriptions
                return self.generate_vocals(f"Sound effect: {description}", "robot", progress_callback)
                
        except Exception as e:
            print(f"❌ Sound effect generation failed: {e}")
            # Fallback to vocals
            return self.generate_vocals(f"Sound effect: {description}", "robot", progress_callback)

class WebMelodyGenerator:
    """Web-optimized wrapper for MelodyGenerator with progress tracking."""
    
    def __init__(self, model_size="medium"):
        self.current_model_size = model_size
        self.generator = MelodyGenerator(model_size=model_size)
        self.lyrics_gen = LyricsGenerator()
        self.vocals_gen = VocalsGenerator(VOCALS_API_KEY)
        self.model_cache = {model_size: self.generator}  # Cache models to avoid reloading
        
    def _get_generator(self, model_size):
        """Get or create a generator for the specified model size."""
        if model_size not in self.model_cache:
            print(f"🔄 Loading {model_size} model...")
            self.model_cache[model_size] = MelodyGenerator(model_size=model_size)
        return self.model_cache[model_size]
        
    def get_device_info(self):
        return self.generator.get_device_info()
    
    def _enhance_genre_prompt(self, genre: str) -> str:
        """Enhance genre descriptions for more professional results."""
        genre_lower = genre.lower()
        
        # Professional enhancement patterns
        enhancements = {
            'drum and bass': 'fast-paced drum and bass with intricate breakbeats, deep sub bass, and jungle-inspired rhythms',
            'dnb': 'energetic drum and bass with complex percussion patterns and heavy bass lines',
            'dubstep': 'heavy dubstep with wobble bass drops, syncopated rhythms, and electronic textures',
            'house': 'uplifting house music with four-on-the-floor beats, piano stabs, and danceable grooves',
            'techno': 'driving techno with repetitive beats, industrial sounds, and hypnotic rhythms',
            'trap': 'modern trap with 808 drum machines, rapid hi-hats, and bass-heavy production',
            'hip hop': 'classic hip hop with rhythmic beats, sampling, and urban groove patterns',
            'lofi': 'relaxing lo-fi hip hop with vintage textures, mellow beats, and nostalgic atmosphere',
            'jazz': 'smooth jazz with improvisation, swing rhythms, and sophisticated harmonies',
            'rock': 'energetic rock music with electric guitars, driving drums, and powerful dynamics',
            'pop': 'catchy pop music with memorable melodies, polished production, and commercial appeal',
            'electronic': 'modern electronic music with synthesizers, digital textures, and innovative sounds',
            'ambient': 'atmospheric ambient music with peaceful soundscapes and meditative qualities',
            'classical': 'elegant classical music with orchestral arrangements and traditional composition',
            'blues': 'soulful blues with twelve-bar progressions, expressive guitar, and emotional depth',
            'reggae': 'laid-back reggae with offbeat rhythms, bass emphasis, and Caribbean vibes',
            'country': 'authentic country music with acoustic instruments, storytelling, and folk traditions',
            'r&b': 'smooth R&B with soulful vocals, groove-based rhythms, and emotional expression',
            'funk': 'groovy funk with syncopated rhythms, bass lines, and danceable energy',
            'synthwave': 'retro synthwave with 80s-inspired synthesizers, nostalgic melodies, and neon aesthetics'
        }
        
        # Check for exact matches first
        for key, enhancement in enhancements.items():
            if key in genre_lower:
                return enhancement
        
        # Fallback: add professional descriptors to the original genre
        return f"professional {genre} music with high-quality production and authentic style"
        
    def generate_song(self, genre_prompt, lyrics_prompt, duration, session_id, model_size="small", include_vocals=False, voice_style="default"):
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
            update_progress(0.05, "Generating lyrics...")
            if lyrics_prompt.strip():
                lyrics = lyrics_prompt
            else:
                # Use enhanced AI lyrics generation
                lyrics = self.lyrics_gen.generate_lyrics(genre_prompt)
            
            # Stage 2: Build song description
            update_progress(0.15, "Building music prompt...")
            
            # Create a more effective prompt for MusicGen
            if lyrics_prompt.strip():
                # If user provided lyrics, create a structured prompt
                music_prompt = f"{genre_prompt} style music with the following theme: {lyrics_prompt.strip()}"
            else:
                # If no lyrics, enhance the genre with professional descriptors
                music_prompt = self._enhance_genre_prompt(genre_prompt)
            
            print(f"🎯 Using enhanced music prompt: '{music_prompt}'")
            
            # Stage 3: Generate melody with specified model size
            base_progress = 0.2
            melody_progress = 0.5 if include_vocals else 0.7
            
            update_progress(base_progress, f"Generating melody with {model_size} model...")
            
            # Get the appropriate generator for the model size
            generator = self._get_generator(model_size)
            
            def progress_callback(progress):
                # Map melody generation progress 
                total_progress = base_progress + (progress * melody_progress)
                update_progress(total_progress, f"Generating melody with {model_size} model...")
            
            melody = generator.generate(
                music_prompt, 
                duration=duration,
                progress_callback=progress_callback
            )
            
            # Stage 4: Generate vocals (if requested)
            vocals_data = None
            if include_vocals and lyrics.strip():
                vocals_start = base_progress + melody_progress
                update_progress(vocals_start, f"Generating vocals with {voice_style} voice...")
                
                def vocals_progress_callback(progress):
                    total_progress = vocals_start + (progress * 0.2)
                    update_progress(total_progress, f"Generating vocals with {voice_style} voice...")
                
                try:
                    vocals_data = self.vocals_gen.generate_vocals(
                        lyrics, 
                        voice_style, 
                        vocals_progress_callback
                    )
                    print("✅ Vocals generation complete")
                except Exception as e:
                    print(f"⚠️ Vocals generation failed: {e}")
                    vocals_data = None
            
            # Stage 5: Save files
            update_progress(0.9, "Saving files...")
            
            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"song_{timestamp}_{session_id[:8]}"
            
            # Save audio
            audio_path = OUTPUT_DIR / f"{filename}.wav"
            generator.save(melody, str(audio_path.with_suffix('')), format="wav")
            
            # Save vocals if generated
            vocals_file = None
            if vocals_data:
                vocals_path = OUTPUT_DIR / f"{filename}_vocals.mp3"
                with open(vocals_path, 'wb') as f:
                    f.write(vocals_data)
                vocals_file = f"{filename}_vocals.mp3"
                print(f"💾 Vocals saved to {vocals_file}")
            
            # Save lyrics
            lyrics_path = OUTPUT_DIR / f"{filename}_lyrics.txt"
            with open(lyrics_path, 'w', encoding='utf-8') as f:
                f.write(f"Genre: {genre_prompt}\n\n")
                f.write(f"Generated Lyrics:\n{lyrics}\n\n")
                f.write(f"Music Prompt: {music_prompt}\n")
                f.write(f"Model Quality: {model_size}\n")
                if include_vocals:
                    f.write(f"Voice Style: {voice_style}\n")
                    f.write(f"Vocals File: {vocals_file or 'Generation failed'}\n")
                f.write(f"Duration: {duration}s\n")
                f.write(f"Device: {generator.device}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
            
            # Complete
            update_progress(1.0, "Generation complete!")
            generation_progress[session_id].update({
                'status': 'complete',
                'audio_file': f"{filename}.wav",
                'vocals_file': vocals_file,
                'lyrics_file': f"{filename}_lyrics.txt",
                'lyrics_content': lyrics
            })
            
            return {
                'success': True,
                'audio_file': f"{filename}.wav",
                'vocals_file': vocals_file,
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

@app.route('/api/generate-lyrics', methods=['POST'])
def generate_ai_lyrics():
    """Generate AI lyrics based on theme and genre."""
    data = request.json
    
    # Validate input
    theme = data.get('theme', '').strip()
    genre = data.get('genre', '').strip()
    
    if not theme and not genre:
        return jsonify({'error': 'Either theme or genre is required'}), 400
    
    # Create session ID
    session_id = str(uuid.uuid4())
    
    # Start generation in background thread
    def generate():
        try:
            generation_progress[session_id] = {
                'status': 'starting',
                'progress': 0,
                'stage': 'Initializing AI lyrics generation...',
                'error': None
            }
            
            def progress_callback(progress):
                stage_messages = {
                    0.1: 'Analyzing theme and genre...',
                    0.2: 'Creating enhanced prompts...',
                    0.3: 'Preparing AI model...',
                    0.5: 'Generating creative lyrics...',
                    0.8: 'Formatting and structuring...',
                    0.9: 'Finalizing lyrics...',
                    1.0: 'AI lyrics complete!'
                }
                
                # Find the appropriate stage message
                stage = 'Generating AI lyrics...'
                for threshold, message in stage_messages.items():
                    if progress >= threshold:
                        stage = message
                
                generation_progress[session_id].update({
                    'progress': int(progress * 100),
                    'stage': stage
                })
            
            # Generate lyrics using enhanced AI
            if theme and genre:
                lyrics = web_generator.lyrics_gen.generate_lyrics(
                    genre, 
                    custom_theme=theme, 
                    progress_callback=progress_callback
                )
            elif genre:
                lyrics = web_generator.lyrics_gen.generate_lyrics(
                    genre, 
                    progress_callback=progress_callback
                )
            else:
                lyrics = web_generator.lyrics_gen.generate_lyrics(
                    theme, 
                    progress_callback=progress_callback
                )
            
            # Save lyrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_lyrics_{timestamp}_{session_id[:8]}.txt"
            file_path = OUTPUT_DIR / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"AI-Generated Lyrics\n")
                f.write(f"Theme: {theme or 'Auto-generated'}\n")
                f.write(f"Genre: {genre or 'Auto-detected'}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write("=" * 50 + "\n\n")
                f.write(lyrics)
            
            generation_progress[session_id].update({
                'status': 'complete',
                'progress': 100,
                'lyrics_content': lyrics,
                'lyrics_file': filename,
                'stage': 'AI lyrics generation complete!'
            })
            
        except Exception as e:
            generation_progress[session_id].update({
                'status': 'error',
                'error': str(e)
            })
    
    thread = threading.Thread(target=generate)
    thread.daemon = True
    thread.start()
    
    return jsonify({'session_id': session_id})

@app.route('/api/generate', methods=['POST'])
def generate_music():
    """Start music generation."""
    data = request.json
    
    # Validate input
    genre_prompt = data.get('genre', '').strip()
    lyrics_prompt = data.get('lyrics', '').strip()
    duration = min(int(data.get('duration', 15)), 60)  # Max 60 seconds for web
    quality = data.get('quality', 'medium')  # Default to medium model for better quality
    include_vocals = data.get('include_vocals', False)  # New vocals option
    voice_style = data.get('voice_style', 'default')  # Voice style option
    
    # Validate quality/model size
    if quality not in ['small', 'medium', 'large']:
        quality = 'small'
        
    # Validate voice style
    if voice_style not in ['default', 'female', 'male', 'robot', 'narrator']:
        voice_style = 'default'
    
    if not genre_prompt:
        return jsonify({'error': 'Genre prompt is required'}), 400
    
    # Create session ID
    session_id = str(uuid.uuid4())
    
    # Start generation in background thread
    def generate():
        web_generator.generate_song(
            genre_prompt, 
            lyrics_prompt, 
            duration, 
            session_id, 
            quality, 
            include_vocals, 
            voice_style
        )
    
    thread = threading.Thread(target=generate)
    thread.daemon = True
    thread.start()
    
    return jsonify({'session_id': session_id})

@app.route('/api/generate-sound-effect', methods=['POST'])
def generate_sound_effect():
    """Generate sound effects."""
    data = request.json
    
    # Validate input
    description = data.get('description', '').strip()
    duration = min(int(data.get('duration', 5)), 10)  # Max 10 seconds for sound effects
    
    if not description:
        return jsonify({'error': 'Sound effect description is required'}), 400
    
    # Create session ID
    session_id = str(uuid.uuid4())
    
    # Start generation in background thread
    def generate():
        try:
            generation_progress[session_id] = {
                'status': 'starting',
                'progress': 0,
                'stage': 'Generating sound effect...',
                'error': None
            }
            
            def progress_callback(progress):
                generation_progress[session_id].update({
                    'progress': int(progress * 100),
                    'stage': 'Generating sound effect...'
                })
            
            # Generate sound effect
            sound_data = web_generator.vocals_gen.generate_sound_effect(
                description, 
                duration, 
                progress_callback
            )
            
            # Save sound effect
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sfx_{timestamp}_{session_id[:8]}.mp3"
            file_path = OUTPUT_DIR / filename
            
            with open(file_path, 'wb') as f:
                f.write(sound_data)
            
            generation_progress[session_id].update({
                'status': 'complete',
                'progress': 100,
                'audio_file': filename,
                'stage': 'Sound effect complete!'
            })
            
        except Exception as e:
            generation_progress[session_id].update({
                'status': 'error',
                'error': str(e)
            })
    
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
    """Get example prompts for the UI with professional descriptions."""
    examples = {
        'genres': [
            "energetic drum and bass with heavy breakbeats",
            "chill lo-fi hip hop with vintage vibes",
            "uplifting house music with piano elements",
            "dark techno with industrial sounds",
            "smooth jazz with saxophone and piano",
            "heavy dubstep with wobble bass drops",
            "classic rock with electric guitar solos",
            "ambient electronic with atmospheric pads",
            "trap beats with 808 drums and hi-hats",
            "reggae with deep bass and offbeat rhythm",
            "synthwave with 80s retro synthesizers",
            "country acoustic with storytelling feel",
            "R&B with soulful groove and vocals",
            "classical piano with orchestral elements",
            "funk with syncopated bass lines"
        ],
        'lyrics_themes': [
            "A motivational anthem about overcoming challenges and reaching your dreams",
            "Romantic ballad about finding true love under starlit skies",
            "High-energy party track about dancing all night with friends",
            "Inspirational song about breaking free from limitations",
            "Nostalgic journey through childhood memories and innocence",
            "Epic adventure song about exploring unknown territories",
            "Peaceful meditation on the beauty of nature and seasons",
            "Celebration of music, creativity, and artistic expression",
            "Urban story about city life and chasing success",
            "Emotional tribute to family bonds and unconditional love",
            "Empowering anthem about self-confidence and inner strength",
            "Reflective piece about growing up and life lessons learned"
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
