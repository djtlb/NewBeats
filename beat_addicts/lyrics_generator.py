import torch
torch.set_default_device('cpu')

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re
import random

class LyricsGenerator:
    def __init__(self, model_name="gpt2"):
        print("🎤 Initializing AI Lyrics Generator...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to('cpu')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = 200  # Enhanced length for better lyrics
        print("✅ Lyrics Generator ready!")
        
        # Genre-specific prompt templates for better lyrics
        self.genre_templates = {
            'hip hop': {
                'intro': "Write rap lyrics about {theme}. Style: confident, rhythmic, street-smart.\n\n[Verse 1]\n",
                'structure': ['verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus'],
                'keywords': ['flow', 'beat', 'rhyme', 'street', 'hustle', 'real', 'struggle', 'rise']
            },
            'pop': {
                'intro': "Write catchy pop song lyrics about {theme}. Style: upbeat, memorable, emotional.\n\n[Verse 1]\n",
                'structure': ['verse', 'pre-chorus', 'chorus', 'verse', 'chorus', 'bridge', 'chorus'],
                'keywords': ['love', 'heart', 'dreams', 'tonight', 'forever', 'dancing', 'feeling', 'shine']
            },
            'rock': {
                'intro': "Write powerful rock lyrics about {theme}. Style: rebellious, energetic, passionate.\n\n[Verse 1]\n",
                'structure': ['verse', 'chorus', 'verse', 'chorus', 'solo', 'bridge', 'chorus'],
                'keywords': ['fire', 'freedom', 'fight', 'loud', 'rebel', 'wild', 'storm', 'thunder']
            },
            'country': {
                'intro': "Write heartfelt country lyrics about {theme}. Style: storytelling, nostalgic, honest.\n\n[Verse 1]\n",
                'structure': ['verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus'],
                'keywords': ['home', 'road', 'heart', 'family', 'memories', 'small town', 'simple', 'true']
            },
            'r&b': {
                'intro': "Write smooth R&B lyrics about {theme}. Style: soulful, romantic, groove-based.\n\n[Verse 1]\n",
                'structure': ['verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus', 'outro'],
                'keywords': ['soul', 'groove', 'smooth', 'baby', 'love', 'feel', 'tonight', 'desire']
            },
            'electronic': {
                'intro': "Write electronic music lyrics about {theme}. Style: futuristic, energetic, hypnotic.\n\n[Verse 1]\n",
                'structure': ['verse', 'drop', 'verse', 'drop', 'breakdown', 'drop'],
                'keywords': ['electric', 'neon', 'digital', 'pulse', 'energy', 'rhythm', 'beat', 'synth']
            },
            'reggae': {
                'intro': "Write reggae lyrics about {theme}. Style: peaceful, conscious, uplifting.\n\n[Verse 1]\n",
                'structure': ['verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus'],
                'keywords': ['peace', 'love', 'unity', 'rhythm', 'island', 'natural', 'positive', 'vibes']
            }
        }
        
        # Common themes for inspiration
        self.lyric_themes = [
            "overcoming challenges and finding strength",
            "celebrating life and good times with friends", 
            "romantic love and deep connection",
            "pursuing dreams and never giving up",
            "nostalgic memories and growing up",
            "freedom and breaking free from limitations",
            "unity and bringing people together",
            "self-confidence and personal empowerment",
            "adventure and exploring new horizons",
            "gratitude and appreciating life's moments"
        ]

    def _detect_genre(self, prompt):
        """Detect genre from prompt text."""
        prompt_lower = prompt.lower()
        
        # Genre detection patterns
        genre_patterns = {
            'hip hop': ['rap', 'hip hop', 'hiphop', 'trap', 'bars', 'flow'],
            'pop': ['pop', 'catchy', 'mainstream', 'radio', 'commercial'],
            'rock': ['rock', 'metal', 'punk', 'guitar', 'heavy'],
            'country': ['country', 'folk', 'acoustic', 'americana', 'nashville'],
            'r&b': ['r&b', 'rnb', 'soul', 'smooth', 'groove'],
            'electronic': ['electronic', 'edm', 'techno', 'house', 'dubstep', 'synth'],
            'reggae': ['reggae', 'ska', 'caribbean', 'jamaica', 'rasta']
        }
        
        for genre, keywords in genre_patterns.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return genre
        
        return 'pop'  # Default genre

    def _create_enhanced_prompt(self, theme, genre=None):
        """Create an enhanced prompt for better lyrics generation."""
        
        if not genre:
            genre = self._detect_genre(theme)
        
        # Get genre template
        template = self.genre_templates.get(genre, self.genre_templates['pop'])
        
        # Create structured prompt
        prompt = template['intro'].format(theme=theme)
        
        # Add some inspiring context
        keywords = template['keywords']
        selected_keywords = random.sample(keywords, min(3, len(keywords)))
        
        prompt += f"Focus on themes like: {', '.join(selected_keywords)}\n"
        prompt += "Create meaningful, original lyrics with good rhythm and flow.\n\n"
        
        return prompt, genre

    def _clean_and_format_lyrics(self, raw_text, genre):
        """Clean and format the generated text into proper lyrics."""
        
        # Remove the original prompt
        lines = raw_text.split('\n')
        cleaned_lines = []
        
        # Find where lyrics actually start
        start_idx = 0
        for i, line in enumerate(lines):
            if '[Verse' in line or line.strip().startswith(('I ', 'You ', 'We ', 'They ', 'She ', 'He ')):
                start_idx = i
                break
        
        # Process lines
        for line in lines[start_idx:]:
            line = line.strip()
            
            # Skip empty lines and prompts
            if not line or line.startswith(('Write ', 'Style:', 'Focus on')):
                continue
                
            # Clean up common AI artifacts
            line = re.sub(r'^[0-9]+\.?\s*', '', line)  # Remove numbering
            line = re.sub(r'\s+', ' ', line)  # Clean whitespace
            
            # Add structure markers if missing
            if len(cleaned_lines) == 0 and not line.startswith('['):
                cleaned_lines.append('[Verse 1]')
            
            cleaned_lines.append(line)
            
            # Limit length for quality
            if len(cleaned_lines) >= 16:  # About 4 verses worth
                break
        
        # Add chorus structure if missing
        formatted_lyrics = self._add_song_structure(cleaned_lines, genre)
        
        return '\n'.join(formatted_lyrics)

    def _add_song_structure(self, lines, genre):
        """Add proper song structure to lyrics."""
        
        structured_lines = []
        verse_count = 1
        line_count = 0
        
        for line in lines:
            if line.startswith('['):
                structured_lines.append(line)
            else:
                structured_lines.append(line)
                line_count += 1
                
                # Add structure every 4 lines
                if line_count % 4 == 0 and line_count < 12:
                    if line_count == 4:
                        structured_lines.append('\n[Chorus]')
                    elif line_count == 8:
                        verse_count += 1
                        structured_lines.append(f'\n[Verse {verse_count}]')
                    elif line_count == 12:
                        structured_lines.append('\n[Chorus]')
        
        return structured_lines

    def generate_lyrics(self, theme_or_genre, custom_theme=None, progress_callback=None):
        """Generate complete AI lyrics with enhanced quality."""
        
        try:
            if progress_callback:
                progress_callback(0.1)
            
            # Determine theme and genre
            if custom_theme:
                theme = custom_theme
                genre = self._detect_genre(theme_or_genre)
            else:
                # If no custom theme, generate one or use a default
                if any(g in theme_or_genre.lower() for g in self.genre_templates.keys()):
                    genre = self._detect_genre(theme_or_genre)
                    theme = random.choice(self.lyric_themes)
                else:
                    theme = theme_or_genre
                    genre = 'pop'
            
            print(f"🎵 Generating {genre} lyrics about: {theme}")
            
            if progress_callback:
                progress_callback(0.2)
            
            # Create enhanced prompt
            enhanced_prompt, detected_genre = self._create_enhanced_prompt(theme, genre)
            
            if progress_callback:
                progress_callback(0.3)
            
            # Generate with better parameters
            inputs = self.tokenizer(enhanced_prompt, return_tensors="pt", truncation=True, max_length=512)
            
            if progress_callback:
                progress_callback(0.5)
            
            # Generate multiple attempts and pick the best
            best_output = None
            best_score = 0
            
            for attempt in range(2):  # Generate 2 attempts, pick best
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=self.max_length + len(inputs.input_ids[0]),
                    temperature=0.8,  # Slightly more creative
                    do_sample=True,
                    top_p=0.9,  # Nucleus sampling for better quality
                    repetition_penalty=1.2,  # Reduce repetition
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                raw_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Simple scoring based on length and variety
                score = len(set(raw_text.split())) / max(len(raw_text.split()), 1)
                
                if score > best_score:
                    best_score = score
                    best_output = raw_text
            
            if progress_callback:
                progress_callback(0.8)
            
            # Clean and format the best output
            formatted_lyrics = self._clean_and_format_lyrics(best_output, detected_genre)
            
            if progress_callback:
                progress_callback(0.9)
            
            # Add metadata
            final_lyrics = f"🎤 AI-Generated {detected_genre.title()} Lyrics\n"
            final_lyrics += f"📝 Theme: {theme}\n\n"
            final_lyrics += formatted_lyrics
            
            if not formatted_lyrics.strip():
                # Fallback if generation failed
                final_lyrics = self._create_fallback_lyrics(theme, detected_genre)
            
            print(f"✅ Generated {len(formatted_lyrics.split())} words of lyrics")
            
            if progress_callback:
                progress_callback(1.0)
            
            return final_lyrics
            
        except Exception as e:
            print(f"❌ Lyrics generation error: {e}")
            return self._create_fallback_lyrics(theme_or_genre, 'pop')

    def _create_fallback_lyrics(self, theme, genre):
        """Create fallback lyrics if AI generation fails."""
        
        fallback_templates = {
            'verse': [
                f"In the rhythm of the night, we find our way",
                f"Through the music and the lights, we come alive", 
                f"Every beat tells a story, every note a dream",
                f"In this moment we are free, nothing's as it seems"
            ],
            'chorus': [
                f"We rise up, we shine bright, like stars in the sky",
                f"No limits, no fears, we're ready to fly",
                f"Together we stand, divided we fall", 
                f"Music unites us, one voice, one call"
            ]
        }
        
        lyrics = f"🎤 AI-Generated {genre.title()} Lyrics\n"
        lyrics += f"📝 Theme: {theme}\n\n"
        lyrics += "[Verse 1]\n"
        lyrics += "\n".join(fallback_templates['verse']) + "\n\n"
        lyrics += "[Chorus]\n" 
        lyrics += "\n".join(fallback_templates['chorus']) + "\n\n"
        lyrics += "[Verse 2]\n"
        lyrics += "The music flows through our souls tonight\n"
        lyrics += "Every moment feels so right\n"
        lyrics += "We dance until the break of dawn\n"
        lyrics += "This feeling will carry on\n\n"
        lyrics += "[Chorus]\n"
        lyrics += "\n".join(fallback_templates['chorus'])
        
        return lyrics

    def generate(self, prompt, max_length=None, progress_callback=None):
        """Legacy method for compatibility."""
        return self.generate_lyrics(prompt, progress_callback=progress_callback)