"""
Multi-language support module for detecting and translating languages.
"""
from typing import Optional, Dict
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

# Set seed for consistent language detection
DetectorFactory.seed = 0


class LanguageManager:
    """Manage multi-language support for the interview agent."""
    
    def __init__(self, default_language: str = 'en', supported_languages: list = None):
        """
        Initialize the language manager.
        
        Args:
            default_language: Default language code (e.g., 'en', 'es')
            supported_languages: List of supported language codes
        """
        self.default_language = default_language
        self.supported_languages = supported_languages or ['en', 'es', 'fr', 'de']
        
        # Language names mapping
        self.language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German'
        }
        
        # Interview prompts in different languages
        self.prompts = {
            'en': {
                'greeting': "Hello! I'm an AI recruiter. I'll be conducting your initial interview today.",
                'ask_name': "Could you please tell me your name?",
                'ask_experience': "How many years of professional experience do you have?",
                'thank_you': "Thank you for your time today. We'll be in touch soon.",
                'clarification': "Could you please clarify that?",
                'next_question': "Let's move on to the next question."
            },
            'es': {
                'greeting': "¡Hola! Soy un reclutador de IA. Conduciré tu entrevista inicial hoy.",
                'ask_name': "¿Podrías decirme tu nombre?",
                'ask_experience': "¿Cuántos años de experiencia profesional tienes?",
                'thank_you': "Gracias por tu tiempo hoy. Estaremos en contacto pronto.",
                'clarification': "¿Podrías aclarar eso?",
                'next_question': "Pasemos a la siguiente pregunta."
            },
            'fr': {
                'greeting': "Bonjour ! Je suis un recruteur IA. Je vais mener votre entretien initial aujourd'hui.",
                'ask_name': "Pourriez-vous me dire votre nom ?",
                'ask_experience': "Combien d'années d'expérience professionnelle avez-vous ?",
                'thank_you': "Merci pour votre temps aujourd'hui. Nous vous contacterons bientôt.",
                'clarification': "Pourriez-vous clarifier cela ?",
                'next_question': "Passons à la question suivante."
            },
            'de': {
                'greeting': "Hallo! Ich bin ein KI-Recruiter. Ich werde heute Ihr erstes Interview führen.",
                'ask_name': "Könnten Sie mir bitte Ihren Namen sagen?",
                'ask_experience': "Wie viele Jahre Berufserfahrung haben Sie?",
                'thank_you': "Vielen Dank für Ihre Zeit heute. Wir werden uns bald melden.",
                'clarification': "Könnten Sie das bitte klären?",
                'next_question': "Kommen wir zur nächsten Frage."
            }
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language code (e.g., 'en', 'es')
        """
        try:
            detected = detect(text)
            return detected
        except Exception:
            return self.default_language
    
    def is_supported(self, language_code: str) -> bool:
        """
        Check if a language is supported.
        
        Args:
            language_code: Language code to check
            
        Returns:
            True if supported, False otherwise
        """
        return language_code in self.supported_languages
    
    def translate_to_english(self, text: str, source_lang: Optional[str] = None) -> str:
        """
        Translate text to English.
        
        Args:
            text: Text to translate
            source_lang: Source language code (auto-detected if None)
            
        Returns:
            Translated text
        """
        if source_lang is None:
            source_lang = self.detect_language(text)
        
        # If already English, return as is
        if source_lang == 'en':
            return text
        
        try:
            translator = GoogleTranslator(source=source_lang, target='en')
            return translator.translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def translate_from_english(self, text: str, target_lang: str) -> str:
        """
        Translate text from English to target language.
        
        Args:
            text: English text to translate
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        # If target is English, return as is
        if target_lang == 'en':
            return text
        
        try:
            translator = GoogleTranslator(source='en', target=target_lang)
            return translator.translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def translate_conversation(
        self,
        conversation: list,
        target_lang: str = 'en'
    ) -> list:
        """
        Translate entire conversation to target language.
        
        Args:
            conversation: List of conversation turns
            target_lang: Target language code
            
        Returns:
            Translated conversation
        """
        translated = []
        
        for turn in conversation:
            translated_turn = turn.copy()
            
            # Detect source language
            source_lang = self.detect_language(turn['content'])
            
            # Translate content
            if source_lang != target_lang:
                try:
                    translator = GoogleTranslator(source=source_lang, target=target_lang)
                    translated_turn['content'] = translator.translate(turn['content'])
                    translated_turn['original_content'] = turn['content']
                    translated_turn['original_language'] = source_lang
                except Exception:
                    # If translation fails, keep original
                    pass
            
            translated.append(translated_turn)
        
        return translated
    
    def get_prompt(self, prompt_key: str, language: Optional[str] = None) -> str:
        """
        Get a localized prompt.
        
        Args:
            prompt_key: Key of the prompt (e.g., 'greeting', 'ask_name')
            language: Language code (uses default if None)
            
        Returns:
            Localized prompt text
        """
        lang = language or self.default_language
        
        # Fall back to English if language not available
        if lang not in self.prompts:
            lang = 'en'
        
        return self.prompts[lang].get(prompt_key, '')
    
    def get_language_name(self, language_code: str) -> str:
        """
        Get the full name of a language from its code.
        
        Args:
            language_code: Language code
            
        Returns:
            Language name
        """
        return self.language_names.get(language_code, language_code.upper())
    
    def add_custom_prompt(self, language: str, key: str, text: str) -> None:
        """
        Add a custom prompt for a language.
        
        Args:
            language: Language code
            key: Prompt key
            text: Prompt text
        """
        if language not in self.prompts:
            self.prompts[language] = {}
        
        self.prompts[language][key] = text
    
    def get_conversation_languages(self, conversation: list) -> Dict[str, int]:
        """
        Analyze languages used in conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Dictionary with language counts
        """
        language_counts = {}
        
        for turn in conversation:
            lang = self.detect_language(turn['content'])
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        return language_counts
    
    def get_dominant_language(self, conversation: list) -> str:
        """
        Get the dominant language in a conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Most frequently used language code
        """
        language_counts = self.get_conversation_languages(conversation)
        
        if not language_counts:
            return self.default_language
        
        return max(language_counts, key=language_counts.get)
