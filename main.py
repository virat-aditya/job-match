from flask import Flask, request, render_template, jsonify
import os
import sqlite3
import docx2txt
import PyPDF2
import re
import nltk
import spacy
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
import logging
import json
import datetime
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load spaCy model (download with: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DATABASE'] = 'skillsync.db'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database for storing historical data and feedback"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Job descriptions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS job_descriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    industry TEXT,
                    experience_level TEXT,
                    skills_required TEXT
                )
            ''')
            
            # Resumes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS resumes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    skills_extracted TEXT,
                    experience_years INTEGER,
                    education_level TEXT
                )
            ''')
            
            # Matching results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS matching_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER,
                    resume_id INTEGER,
                    similarity_score REAL,
                    semantic_score REAL,
                    skills_score REAL,
                    experience_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES job_descriptions (id),
                    FOREIGN KEY (resume_id) REFERENCES resumes (id)
                )
            ''')
            
            # Feedback table for learning
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hiring_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER,
                    resume_id INTEGER,
                    hired BOOLEAN,
                    interview_called BOOLEAN,
                    feedback_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES job_descriptions (id),
                    FOREIGN KEY (resume_id) REFERENCES resumes (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def store_job_description(self, content, industry=None, experience_level=None, skills_required=None):
        """Store job description and return ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO job_descriptions (content, industry, experience_level, skills_required)
                VALUES (?, ?, ?, ?)
            ''', (content, industry, experience_level, json.dumps(skills_required) if skills_required else None))
            
            job_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return job_id
            
        except Exception as e:
            logger.error(f"Error storing job description: {str(e)}")
            return None
    
    def store_resume(self, filename, content, skills_extracted=None, experience_years=None, education_level=None):
        """Store resume and return ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO resumes (filename, content, skills_extracted, experience_years, education_level)
                VALUES (?, ?, ?, ?, ?)
            ''', (filename, content, json.dumps(skills_extracted) if skills_extracted else None, 
                  experience_years, education_level))
            
            resume_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return resume_id
            
        except Exception as e:
            logger.error(f"Error storing resume: {str(e)}")
            return None
    
    def store_matching_result(self, job_id, resume_id, similarity_score, semantic_score, skills_score, experience_score):
        """Store matching result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO matching_results (job_id, resume_id, similarity_score, semantic_score, skills_score, experience_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (job_id, resume_id, similarity_score, semantic_score, skills_score, experience_score))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing matching result: {str(e)}")


class AdvancedNLPProcessor:
    """Advanced NLP processing with NER, semantic analysis, and feature extraction"""
    
    def __init__(self):
        self.nlp = nlp
        
        # Enhanced skill categories with more comprehensive keywords
        self.skill_categories = {
            'programming_languages': {
                'keywords': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust', 'kotlin', 'swift'],
                'weight': 1.5
            },
            'web_technologies': {
                'keywords': ['html', 'css', 'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring'],
                'weight': 1.3
            },
            'databases': {
                'keywords': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'oracle'],
                'weight': 1.4
            },
            'data_science': {
                'keywords': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'keras', 'opencv'],
                'weight': 1.6
            },
            'cloud_platforms': {
                'keywords': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'ci/cd'],
                'weight': 1.4
            },
            'soft_skills': {
                'keywords': ['leadership', 'communication', 'teamwork', 'problem solving', 'analytical', 'creative'],
                'weight': 1.1
            },
            'methodologies': {
                'keywords': ['agile', 'scrum', 'kanban', 'devops', 'test driven development', 'microservices'],
                'weight': 1.2
            }
        }
        
        # Experience level patterns
        self.experience_patterns = {
            'years': [
                r'(\d+)[\+\s]*years?\s+(?:of\s+)?experience',
                r'(\d+)[\+\s]*yrs?\s+(?:of\s+)?experience',
                r'experience[:\s]*(\d+)[\+\s]*years?',
                r'(\d+)[\+\s]*years?\s+in',
                r'over\s+(\d+)\s+years?'
            ],
            'seniority': {
                'entry': ['entry level', 'junior', 'associate', 'trainee', 'intern', 'graduate'],
                'mid': ['mid level', 'intermediate', 'experienced', 'professional'],
                'senior': ['senior', 'lead', 'principal', 'architect', 'manager', 'director'],
                'executive': ['vp', 'vice president', 'cto', 'ceo', 'head of', 'chief']
            }
        }
        
        # Education patterns
        self.education_patterns = {
            'degree_types': ['bachelor', 'master', 'phd', 'doctorate', 'associates', 'diploma', 'certificate'],
            'fields': ['computer science', 'engineering', 'mathematics', 'statistics', 'physics', 'business']
        }
    
    def extract_named_entities(self, text: str) -> Dict:
        """Extract named entities using spaCy"""
        if not self.nlp:
            return {}
        
        try:
            doc = self.nlp(text)
            entities = {
                'organizations': [],
                'skills': [],
                'locations': [],
                'dates': [],
                'technologies': []
            }
            
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    entities['organizations'].append(ent.text)
                elif ent.label_ == "GPE":
                    entities['locations'].append(ent.text)
                elif ent.label_ == "DATE":
                    entities['dates'].append(ent.text)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in NER extraction: {str(e)}")
            return {}
    
    def extract_skills_advanced(self, text: str) -> Dict:
        """Advanced skill extraction with categorization and confidence scoring"""
        try:
            text_lower = text.lower()
            skills_found = defaultdict(list)
            skill_confidence = {}
            
            for category, category_data in self.skill_categories.items():
                keywords = category_data['keywords']
                weight = category_data['weight']
                
                for keyword in keywords:
                    # Count occurrences and context
                    pattern = rf'\b{re.escape(keyword)}\b'
                    matches = re.findall(pattern, text_lower)
                    
                    if matches:
                        count = len(matches)
                        confidence = min(count * 0.2, 1.0) * weight
                        
                        skills_found[category].append({
                            'skill': keyword,
                            'count': count,
                            'confidence': confidence
                        })
                        skill_confidence[keyword] = confidence
            
            return {
                'categorized_skills': dict(skills_found),
                'skill_confidence': skill_confidence,
                'total_skills': len(skill_confidence)
            }
            
        except Exception as e:
            logger.error(f"Error in advanced skill extraction: {str(e)}")
            return {'categorized_skills': {}, 'skill_confidence': {}, 'total_skills': 0}
    
    def extract_experience_level(self, text: str) -> Dict:
        """Extract experience level information"""
        try:
            text_lower = text.lower()
            experience_info = {
                'years': 0,
                'level': 'entry',
                'confidence': 0.0
            }
            
            # Extract years of experience
            for pattern in self.experience_patterns['years']:
                matches = re.findall(pattern, text_lower)
                if matches:
                    years = max([int(match) for match in matches])
                    experience_info['years'] = years
                    break
            
            # Determine seniority level
            max_score = 0
            detected_level = 'entry'
            
            for level, keywords in self.experience_patterns['seniority'].items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > max_score:
                    max_score = score
                    detected_level = level
            
            experience_info['level'] = detected_level
            experience_info['confidence'] = min(max_score * 0.3, 1.0)
            
            # Adjust level based on years
            if experience_info['years'] >= 8:
                experience_info['level'] = 'senior'
            elif experience_info['years'] >= 4:
                experience_info['level'] = 'mid'
            elif experience_info['years'] >= 1:
                experience_info['level'] = 'entry'
            
            return experience_info
            
        except Exception as e:
            logger.error(f"Error extracting experience level: {str(e)}")
            return {'years': 0, 'level': 'entry', 'confidence': 0.0}
    
    def extract_education_info(self, text: str) -> Dict:
        """Extract education information"""
        try:
            text_lower = text.lower()
            education_info = {
                'highest_degree': 'none',
                'field_of_study': [],
                'institutions': [],
                'confidence': 0.0
            }
            
            # Extract degree types
            degree_scores = {}
            for degree in self.education_patterns['degree_types']:
                if degree in text_lower:
                    degree_scores[degree] = text_lower.count(degree)
            
            if degree_scores:
                highest_degree = max(degree_scores.keys(), key=lambda x: degree_scores[x])
                education_info['highest_degree'] = highest_degree
                education_info['confidence'] = min(degree_scores[highest_degree] * 0.4, 1.0)
            
            # Extract fields of study
            for field in self.education_patterns['fields']:
                if field in text_lower:
                    education_info['field_of_study'].append(field)
            
            # Extract institutions using NER
            if self.nlp:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "ORG" and any(word in ent.text.lower() for word in ['university', 'college', 'institute', 'school']):
                        education_info['institutions'].append(ent.text)
            
            return education_info
            
        except Exception as e:
            logger.error(f"Error extracting education info: {str(e)}")
            return {'highest_degree': 'none', 'field_of_study': [], 'institutions': [], 'confidence': 0.0}


class SemanticMatcher:
    """Advanced semantic matching using transformer models"""
    
    def __init__(self):
        try:
            # Load pre-trained sentence transformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.model_loaded = True
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentence transformer: {str(e)}")
            self.sentence_model = None
            self.model_loaded = False
    
    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        try:
            if not self.model_loaded or not text1 or not text2:
                return 0.0
            
            # Generate embeddings
            embeddings = self.sentence_model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
    
    def get_section_similarities(self, job_desc: str, resume_text: str) -> Dict:
        """Calculate similarities for different sections"""
        try:
            # Split into sections (basic approach)
            job_sections = self._split_into_sections(job_desc)
            resume_sections = self._split_into_sections(resume_text)
            
            section_similarities = {}
            
            for job_section, job_content in job_sections.items():
                if job_content and job_section in resume_sections and resume_sections[job_section]:
                    similarity = self.get_semantic_similarity(job_content, resume_sections[job_section])
                    section_similarities[job_section] = similarity
            
            return section_similarities
            
        except Exception as e:
            logger.error(f"Error calculating section similarities: {str(e)}")
            return {}
    
    def _split_into_sections(self, text: str) -> Dict:
        """Basic section splitting (can be enhanced with more sophisticated methods)"""
        sections = {
            'skills': '',
            'experience': '',
            'education': '',
            'summary': ''
        }
        
        text_lower = text.lower()
        
        # Simple keyword-based section detection
        if any(word in text_lower for word in ['skills', 'technical', 'technologies']):
            sections['skills'] = text[:200]  # First 200 chars as proxy
        
        if any(word in text_lower for word in ['experience', 'work', 'employment']):
            sections['experience'] = text[:300]
        
        if any(word in text_lower for word in ['education', 'degree', 'university']):
            sections['education'] = text[:200]
        
        sections['summary'] = text[:150]  # First part as summary
        
        return sections


class MLEnhancedMatcher:
    """Machine Learning enhanced matching with clustering and classification"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.clusterer = KMeans(n_clusters=5, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        
        # Try to load pre-trained models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            if os.path.exists('models/classifier.pkl'):
                with open('models/classifier.pkl', 'rb') as f:
                    self.classifier = pickle.load(f)
                with open('models/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info("Pre-trained models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def _save_models(self):
        """Save trained models"""
        try:
            os.makedirs('models', exist_ok=True)
            with open('models/classifier.pkl', 'wb') as f:
                pickle.dump(self.classifier, f)
            with open('models/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def extract_ml_features(self, job_text: str, resume_text: str, skills_data: Dict, experience_data: Dict, education_data: Dict) -> np.ndarray:
        """Extract comprehensive features for ML models"""
        try:
            features = []
            
            # Basic text features
            features.extend([
                len(job_text.split()),
                len(resume_text.split()),
                len(set(job_text.lower().split()) & set(resume_text.lower().split())),
            ])
            
            # Skills features
            total_skills = skills_data.get('total_skills', 0)
            avg_confidence = np.mean(list(skills_data.get('skill_confidence', {0: 0}).values())) if skills_data.get('skill_confidence') else 0
            features.extend([total_skills, avg_confidence])
            
            # Experience features
            features.extend([
                experience_data.get('years', 0),
                1 if experience_data.get('level') == 'senior' else 0,
                1 if experience_data.get('level') == 'mid' else 0,
                experience_data.get('confidence', 0)
            ])
            
            # Education features
            education_score = {
                'phd': 5, 'doctorate': 5,
                'master': 4,
                'bachelor': 3,
                'associates': 2,
                'diploma': 1,
                'certificate': 1,
                'none': 0
            }.get(education_data.get('highest_degree', 'none'), 0)
            
            features.extend([
                education_score,
                len(education_data.get('field_of_study', [])),
                education_data.get('confidence', 0)
            ])
            
            # Text similarity features (using traditional methods as backup)
            vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
            try:
                tfidf_matrix = vectorizer.fit_transform([job_text, resume_text])
                tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                tfidf_sim = 0.0
            
            features.append(tfidf_sim)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting ML features: {str(e)}")
            return np.zeros((1, 14))  # Return zero features as fallback
    
    def predict_match_probability(self, features: np.ndarray) -> float:
        """Predict match probability using trained classifier"""
        try:
            if not self.is_trained:
                # Return basic score if not trained
                return float(features[0][-1])  # Use TF-IDF similarity as fallback
            
            scaled_features = self.scaler.transform(features)
            probabilities = self.classifier.predict_proba(scaled_features)
            
            # Return probability of positive match
            return float(probabilities[0][1]) if probabilities.shape[1] > 1 else float(probabilities[0][0])
            
        except Exception as e:
            logger.error(f"Error predicting match probability: {str(e)}")
            return 0.0
    
    def detect_anomalies(self, features_list: List[np.ndarray]) -> List[bool]:
        """Detect anomalous resumes"""
        try:
            if len(features_list) < 2:
                return [False] * len(features_list)
            
            combined_features = np.vstack(features_list)
            anomaly_scores = self.anomaly_detector.fit_predict(combined_features)
            
            return [score == -1 for score in anomaly_scores]
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return [False] * len(features_list)


class AdvancedResumeJobMatcher:
    """Main enhanced matcher class combining all advanced features"""
    
    def __init__(self):
        self.nlp_processor = AdvancedNLPProcessor()
        self.semantic_matcher = SemanticMatcher()
        self.ml_matcher = MLEnhancedMatcher()
        self.db_manager = DatabaseManager(app.config['DATABASE'])
        
        # A/B testing configuration
        self.ab_test_config = {
            'use_semantic': True,
            'use_ml_features': True,
            'use_anomaly_detection': True
        }
    
    def comprehensive_analysis(self, job_description: str, resume_text: str) -> Dict:
        """Perform comprehensive analysis of job-resume match"""
        try:
            # Extract entities and features
            job_entities = self.nlp_processor.extract_named_entities(job_description)
            resume_entities = self.nlp_processor.extract_named_entities(resume_text)
            
            # Skills analysis
            job_skills = self.nlp_processor.extract_skills_advanced(job_description)
            resume_skills = self.nlp_processor.extract_skills_advanced(resume_text)
            
            # Experience analysis
            job_experience = self.nlp_processor.extract_experience_level(job_description)
            resume_experience = self.nlp_processor.extract_experience_level(resume_text)
            
            # Education analysis
            resume_education = self.nlp_processor.extract_education_info(resume_text)
            
            # Semantic similarity
            semantic_score = 0.0
            if self.ab_test_config['use_semantic']:
                semantic_score = self.semantic_matcher.get_semantic_similarity(job_description, resume_text)
                section_similarities = self.semantic_matcher.get_section_similarities(job_description, resume_text)
            else:
                section_similarities = {}
            
            # Skills matching score
            skills_score = self._calculate_skills_match_score(job_skills, resume_skills)
            
            # Experience matching score
            experience_score = self._calculate_experience_match_score(job_experience, resume_experience)
            
            # Education matching score
            education_score = self._calculate_education_match_score(resume_education)
            
            # ML features and prediction
            ml_features = self.ml_matcher.extract_ml_features(
                job_description, resume_text, resume_skills, resume_experience, resume_education
            )
            
            ml_probability = 0.0
            if self.ab_test_config['use_ml_features']:
                ml_probability = self.ml_matcher.predict_match_probability(ml_features)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                semantic_score, skills_score, experience_score, education_score, ml_probability
            )
            
            return {
                'composite_score': composite_score,
                'semantic_score': semantic_score,
                'skills_score': skills_score,
                'experience_score': experience_score,
                'education_score': education_score,
                'ml_probability': ml_probability,
                'job_entities': job_entities,
                'resume_entities': resume_entities,
                'job_skills': job_skills,
                'resume_skills': resume_skills,
                'resume_experience': resume_experience,
                'resume_education': resume_education,
                'section_similarities': section_similarities,
                'ml_features': ml_features.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            return self._get_fallback_analysis()
    
    def _calculate_skills_match_score(self, job_skills: Dict, resume_skills: Dict) -> float:
        """Calculate skills matching score with advanced weighting"""
        try:
            job_skill_set = set(job_skills.get('skill_confidence', {}).keys())
            resume_skill_set = set(resume_skills.get('skill_confidence', {}).keys())
            
            if not job_skill_set:
                return 0.0
            
            # Calculate weighted intersection
            common_skills = job_skill_set & resume_skill_set
            
            if not common_skills:
                return 0.0
            
            # Weight by confidence and importance
            total_weight = 0
            matched_weight = 0
            
            for skill in job_skill_set:
                weight = job_skills['skill_confidence'].get(skill, 0.5)
                total_weight += weight
                
                if skill in common_skills:
                    resume_confidence = resume_skills['skill_confidence'].get(skill, 0.5)
                    matched_weight += min(weight, resume_confidence)
            
            return matched_weight / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating skills match score: {str(e)}")
            return 0.0
    
    def _calculate_experience_match_score(self, job_exp: Dict, resume_exp: Dict) -> float:
        """Calculate experience matching score"""
        try:
            # Years experience matching
            job_years = job_exp.get('years', 0)
            resume_years = resume_exp.get('years', 0)
            
            if job_years == 0:
                years_score = 1.0  # No specific requirement
            else:
                years_ratio = min(resume_years / job_years, 1.0)
                years_score = max(0.0, years_ratio)
            
            # Level matching
            level_mapping = {'entry': 1, 'mid': 2, 'senior': 3, 'executive': 4}
            job_level = level_mapping.get(job_exp.get('level', 'entry'), 1)
            resume_level = level_mapping.get(resume_exp.get('level', 'entry'), 1)
            
            level_score = 1.0 - abs(job_level - resume_level) / 4.0
            level_score = max(0.0, level_score)
            
            # Combine scores
            return (years_score * 0.6 + level_score * 0.4)
            
        except Exception as e:
            logger.error(f"Error calculating experience match score: {str(e)}")
            return 0.0
    
    def _calculate_education_match_score(self, resume_education: Dict) -> float:
        """Calculate education score (higher education = higher score)"""
        try:
            degree_scores = {
                'phd': 1.0, 'doctorate': 1.0,
                'master': 0.8,
                'bachelor': 0.6,
                'associates': 0.4,
                'diploma': 0.3,
                'certificate': 0.2,
                'none': 0.0
            }
            
            degree = resume_education.get('highest_degree', 'none')
            base_score = degree_scores.get(degree, 0.0)
            
            # Bonus for relevant field of study
            relevant_fields = ['computer science', 'engineering', 'mathematics', 'statistics']
            field_bonus = 0.1 if any(field in resume_education.get('field_of_study', []) for field in relevant_fields) else 0.0
            
            return min(base_score + field_bonus, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating education match score: {str(e)}")
            return 0.0
    
    def _calculate_composite_score(self, semantic_score: float, skills_score: float, 
                                 experience_score: float, education_score: float, ml_probability: float) -> float:
        """Calculate weighted composite score"""
        try:
            # Weights for different components
            weights = {
                'semantic': 0.3,
                'skills': 0.35,
                'experience': 0.2,
                'education': 0.1,
                'ml': 0.05
            }
            
            composite = (
                semantic_score * weights['semantic'] +
                skills_score * weights['skills'] +
                experience_score * weights['experience'] +
                education_score * weights['education'] +
                ml_probability * weights['ml']
            )
            
            return min(max(composite, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {str(e)}")
            return 0.0
    
    def _get_fallback_analysis(self) -> Dict:
        """Return fallback analysis in case of errors"""
        return {
            'composite_score': 0.0,
            'semantic_score': 0.0,
            'skills_score': 0.0,
            'experience_score': 0.0,
            'education_score': 0.0,
            'ml_probability': 0.0,
            'job_entities': {},
            'resume_entities': {},
            'job_skills': {},
            'resume_skills': {},
            'resume_experience': {},
            'resume_education': {},
            'section_similarities': {},
            'ml_features': []
        }
    
    def batch_analyze_resumes(self, job_description: str, resume_data_list: List[Dict]) -> List[Dict]:
        """Analyze multiple resumes with advanced features"""
        try:
            results = []
            all_features = []
            
            # First pass: analyze all resumes and collect features
            for resume_data in resume_data_list:
                analysis = self.comprehensive_analysis(job_description, resume_data['text'])
                analysis['filename'] = resume_data['filename']
                results.append(analysis)
                all_features.append(analysis['ml_features'])
            
            # Anomaly detection
            if self.ab_test_config['use_anomaly_detection'] and len(all_features) > 1:
                feature_arrays = [np.array(features).reshape(1, -1) for features in all_features]
                anomaly_flags = self.ml_matcher.detect_anomalies(feature_arrays)
                
                for i, result in enumerate(results):
                    result['is_anomaly'] = anomaly_flags[i] if i < len(anomaly_flags) else False
            else:
                for result in results:
                    result['is_anomaly'] = False
            
            # Sort by composite score
            results.sort(key=lambda x: x['composite_score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}")
            return []
    
    def get_analytics_data(self, results: List[Dict]) -> Dict:
        """Generate analytics data for visualization"""
        try:
            if not results:
                return {}
            
            analytics = {
                'score_distribution': [r['composite_score'] for r in results],
                'skills_analysis': {},
                'experience_distribution': {},
                'education_distribution': {},
                'top_skills': {},
                'anomaly_count': sum(1 for r in results if r.get('is_anomaly', False))
            }
            
            # Skills analysis
            all_skills = defaultdict(int)
            for result in results:
                for skill, confidence in result.get('resume_skills', {}).get('skill_confidence', {}).items():
                    all_skills[skill] += confidence
            
            analytics['top_skills'] = dict(sorted(all_skills.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Experience distribution
            exp_levels = [r.get('resume_experience', {}).get('level', 'entry') for r in results]
            analytics['experience_distribution'] = dict(Counter(exp_levels))
            
            # Education distribution
            edu_levels = [r.get('resume_education', {}).get('highest_degree', 'none') for r in results]
            analytics['education_distribution'] = dict(Counter(edu_levels))
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error generating analytics data: {str(e)}")
            return {}


class FeedbackLearningSystem:
    """System for learning from hiring feedback"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def collect_feedback(self, job_id: int, resume_id: int, hired: bool, interview_called: bool = False):
        """Collect hiring feedback for model improvement"""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO hiring_feedback (job_id, resume_id, hired, interview_called)
                VALUES (?, ?, ?, ?)
            ''', (job_id, resume_id, hired, interview_called))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Feedback collected: job_id={job_id}, resume_id={resume_id}, hired={hired}")
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {str(e)}")
    
    def retrain_model_with_feedback(self):
        """Retrain ML models using collected feedback"""
        try:
            # This would implement periodic retraining
            # For now, just log the intent
            logger.info("Model retraining with feedback - placeholder implementation")
            
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}")


# Initialize components
enhanced_matcher = AdvancedResumeJobMatcher()
feedback_system = FeedbackLearningSystem(enhanced_matcher.db_manager)


def extract_text_from_pdf(file_path):
    """Extract text from PDF with enhanced error handling"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                try:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + " "
                except Exception as page_error:
                    logger.warning(f"Error extracting page {page_num} from PDF: {str(page_error)}")
                    continue
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF text from {file_path}: {str(e)}")
        return ""


def extract_text_from_docx(file_path):
    """Extract text from DOCX with enhanced error handling"""
    try:
        text = docx2txt.process(file_path)
        return text.strip() if text else ""
    except Exception as e:
        logger.error(f"Error extracting DOCX text from {file_path}: {str(e)}")
        return ""


def extract_text_from_txt(file_path):
    """Extract text from TXT with multiple encoding support"""
    try:
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read().strip()
            except UnicodeDecodeError:
                continue
        logger.warning(f"Could not decode text file: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting TXT text from {file_path}: {str(e)}")
        return ""


def extract_text(file_path):
    """Enhanced text extraction with comprehensive error handling"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ""

        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            return extract_text_from_txt(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return ""
            
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""


@app.route("/")
def skillsync():
    return render_template('skillsync.html')


@app.route('/matcher', methods=['POST'])
def matcher_route():
    """Enhanced matching route with comprehensive analysis"""
    try:
        if request.method == 'POST':
            job_description = request.form.get('job_description', '').strip()
            resume_files = request.files.getlist('resumes')

            # Enhanced validation
            if not job_description or len(job_description) < 50:
                return render_template('skillsync.html',
                                     message="Please enter a detailed job description (at least 50 characters).",
                                     error=True)

            if not resume_files or not any(file.filename for file in resume_files):
                return render_template('skillsync.html',
                                     message="Please upload at least one resume file.",
                                     error=True)

            # Store job description in database
            job_id = enhanced_matcher.db_manager.store_job_description(job_description)

            # Process resumes with enhanced analysis
            resume_data_list = []
            valid_files = 0

            for resume_file in resume_files:
                if resume_file.filename:
                    try:
                        # Secure filename handling
                        filename = re.sub(r'[^\w\s.-]', '', resume_file.filename)
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                        # Save and extract text
                        resume_file.save(file_path)
                        resume_text = extract_text(file_path)

                        if resume_text and len(resume_text.strip()) > 50:  # Minimum content check
                            resume_data_list.append({
                                'filename': resume_file.filename,
                                'text': resume_text,
                                'file_path': file_path
                            })
                            valid_files += 1
                        else:
                            logger.warning(f"Insufficient content in {resume_file.filename}")

                        # Clean up file
                        try:
                            os.remove(file_path)
                        except:
                            pass

                    except Exception as e:
                        logger.error(f"Error processing file {resume_file.filename}: {str(e)}")
                        continue

            if valid_files == 0:
                return render_template('skillsync.html',
                                     message="No valid resume content could be extracted. Please ensure files contain readable text.",
                                     error=True)

            # Perform comprehensive analysis
            logger.info(f"Starting comprehensive analysis for {valid_files} resumes")
            analysis_results = enhanced_matcher.batch_analyze_resumes(job_description, resume_data_list)

            if not analysis_results:
                return render_template('skillsync.html',
                                     message="Could not analyze resumes. Please try again.",
                                     error=True)

            # Store results in database
            for i, result in enumerate(analysis_results):
                try:
                    # Store resume in database
                    resume_id = enhanced_matcher.db_manager.store_resume(
                        result['filename'],
                        resume_data_list[i]['text'],
                        result.get('resume_skills'),
                        result.get('resume_experience', {}).get('years'),
                        result.get('resume_education', {}).get('highest_degree')
                    )
                    
                    # Store matching result
                    if job_id and resume_id:
                        enhanced_matcher.db_manager.store_matching_result(
                            job_id, resume_id,
                            result['composite_score'],
                            result['semantic_score'],
                            result['skills_score'],
                            result['experience_score']
                        )
                        
                except Exception as e:
                    logger.error(f"Error storing result for {result['filename']}: {str(e)}")

            # Prepare results for display
            display_results = []
            for result in analysis_results[:10]:  # Top 10 results
                score_percentage = round(result['composite_score'] * 100)
                
                # Create detailed breakdown
                breakdown = {
                    'semantic': round(result['semantic_score'] * 100),
                    'skills': round(result['skills_score'] * 100),
                    'experience': round(result['experience_score'] * 100),
                    'education': round(result['education_score'] * 100),
                    'ml_prediction': round(result['ml_probability'] * 100)
                }
                
                display_results.append({
                    'filename': result['filename'],
                    'score': score_percentage,
                    'breakdown': breakdown,
                    'is_anomaly': result.get('is_anomaly', False),
                    'total_skills': result.get('resume_skills', {}).get('total_skills', 0),
                    'experience_years': result.get('resume_experience', {}).get('years', 0),
                    'education_level': result.get('resume_education', {}).get('highest_degree', 'none')
                })

            # Generate analytics
            analytics = enhanced_matcher.get_analytics_data(analysis_results)

            return render_template('enhanced_skillsync.html',
                                 message=f"Advanced analysis completed for {valid_files} resumes",
                                 results=display_results,
                                 analytics=analytics,
                                 job_description=job_description,
                                 total_processed=valid_files)

    except Exception as e:
        logger.error(f"Unexpected error in enhanced matcher route: {str(e)}")
        return render_template('skillsync.html',
                             message="An unexpected error occurred during analysis. Please try again.",
                             error=True)


@app.route('/analytics')
def analytics_dashboard():
    """Analytics dashboard route"""
    try:
        # Get historical data for analytics
        conn = sqlite3.connect(app.config['DATABASE'])
        
        # Recent matching trends
        df_matches = pd.read_sql_query('''
            SELECT DATE(created_at) as date, AVG(similarity_score) as avg_score, COUNT(*) as count
            FROM matching_results 
            WHERE created_at >= date('now', '-30 days')
            GROUP BY DATE(created_at)
            ORDER BY date
        ''', conn)
        
        # Skills trends
        df_skills = pd.read_sql_query('''
            SELECT skills_extracted, COUNT(*) as frequency
            FROM resumes 
            WHERE skills_extracted IS NOT NULL AND created_at >= date('now', '-30 days')
            GROUP BY skills_extracted
            ORDER BY frequency DESC
            LIMIT 20
        ''', conn)
        
        conn.close()
        
        analytics_data = {
            'matching_trends': df_matches.to_dict('records') if not df_matches.empty else [],
            'skills_trends': df_skills.to_dict('records') if not df_skills.empty else [],
            'total_jobs': len(df_matches),
            'total_resumes': df_skills['frequency'].sum() if not df_skills.empty else 0
        }
        
        return render_template('analytics_dashboard.html', analytics=analytics_data)
        
    except Exception as e:
        logger.error(f"Error in analytics dashboard: {str(e)}")
        return render_template('analytics_dashboard.html', analytics={})


@app.route('/feedback', methods=['POST'])
def feedback_route():
    """Collect hiring feedback for model improvement"""
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        resume_id = data.get('resume_id')
        hired = data.get('hired', False)
        interview_called = data.get('interview_called', False)
        
        if job_id and resume_id:
            feedback_system.collect_feedback(job_id, resume_id, hired, interview_called)
            return jsonify({'status': 'success', 'message': 'Feedback collected successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Missing required parameters'})
            
    except Exception as e:
        logger.error(f"Error collecting feedback: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Error collecting feedback'})


@app.route('/api/skills-gap-analysis', methods=['POST'])
def skills_gap_analysis():
    """API endpoint for detailed skills gap analysis"""
    try:
        data = request.get_json()
        job_description = data.get('job_description', '')
        resume_text = data.get('resume_text', '')
        
        if not job_description or not resume_text:
            return jsonify({'error': 'Missing job description or resume text'})
        
        # Perform detailed skills analysis
        job_skills = enhanced_matcher.nlp_processor.extract_skills_advanced(job_description)
        resume_skills = enhanced_matcher.nlp_processor.extract_skills_advanced(resume_text)
        
        # Calculate gaps
        job_skill_set = set(job_skills.get('skill_confidence', {}).keys())
        resume_skill_set = set(resume_skills.get('skill_confidence', {}).keys())
        
        gap_analysis = {
            'required_skills': list(job_skill_set),
            'candidate_skills': list(resume_skill_set),
            'matching_skills': list(job_skill_set & resume_skill_set),
            'missing_skills': list(job_skill_set - resume_skill_set),
            'additional_skills': list(resume_skill_set - job_skill_set),
            'match_percentage': len(job_skill_set & resume_skill_set) / len(job_skill_set) * 100 if job_skill_set else 0
        }
        
        return jsonify(gap_analysis)
        
    except Exception as e:
        logger.error(f"Error in skills gap analysis: {str(e)}")
        return jsonify({'error': 'Analysis failed'})


@app.route('/api/resume-quality-score', methods=['POST'])
def resume_quality_score():
    """API endpoint for resume quality scoring"""
    try:
        data = request.get_json()
        resume_text = data.get('resume_text', '')
        
        if not resume_text:
            return jsonify({'error': 'Missing resume text'})
        
        # Calculate quality metrics
        word_count = len(resume_text.split())
        sentence_count = len(re.split(r'[.!?]+', resume_text))
        
        # Skills diversity
        skills_data = enhanced_matcher.nlp_processor.extract_skills_advanced(resume_text)
        skills_diversity = len(skills_data.get('categorized_skills', {}))
        
        # Experience clarity
        experience_data = enhanced_matcher.nlp_processor.extract_experience_level(resume_text)
        experience_clarity = experience_data.get('confidence', 0.0)
        
        # Education completeness
        education_data = enhanced_matcher.nlp_processor.extract_education_info(resume_text)
        education_completeness = education_data.get('confidence', 0.0)
        
        # Calculate overall quality score
        quality_metrics = {
            'word_count_score': min(word_count / 500, 1.0),  # Optimal around 500 words
            'skills_diversity_score': min(skills_diversity / 10, 1.0),  # Good diversity around 10 categories
            'experience_clarity_score': experience_clarity,
            'education_completeness_score': education_completeness,
            'structure_score': min(sentence_count / word_count * 10, 1.0) if word_count > 0 else 0  # Good sentence structure
        }
        
        overall_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        return jsonify({
            'overall_quality_score': round(overall_score * 100),
            'metrics': {k: round(v * 100) for k, v in quality_metrics.items()},
            'recommendations': get_quality_recommendations(quality_metrics)
        })
        
    except Exception as e:
        logger.error(f"Error in resume quality scoring: {str(e)}")
        return jsonify({'error': 'Quality scoring failed'})


def get_quality_recommendations(metrics: Dict) -> List[str]:
    """Generate recommendations based on quality metrics"""
    recommendations = []
    
    if metrics['word_count_score'] < 0.5:
        recommendations.append("Consider adding more detail to your resume (aim for 300-600 words)")
    
    if metrics['skills_diversity_score'] < 0.5:
        recommendations.append("Include more diverse technical and soft skills")
    
    if metrics['experience_clarity_score'] < 0.6:
        recommendations.append("Clearly state your years of experience and specific roles")
    
    if metrics['education_completeness_score'] < 0.6:
        recommendations.append("Include complete education information with degrees and institutions")
    
    if metrics['structure_score'] < 0.5:
        recommendations.append("Improve resume structure with clear sections and bullet points")
    
    if not recommendations:
        recommendations.append("Your resume shows good quality across all metrics!")
    
    return recommendations


@app.errorhandler(413)
def too_large(e):
    return render_template('skillsync.html',
                         message="File too large. Please upload files smaller than 16MB.",
                         error=True)


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return render_template('skillsync.html',
                         message="Internal server error. Our team has been notified.",
                         error=True)


if __name__ == '__main__':
    try:
        # Create necessary directories
        for directory in [app.config['UPLOAD_FOLDER'], 'models', 'data']:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")

        # Set maximum file size (16MB)
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

        logger.info("Starting Enhanced SkillSync with Advanced Data Science Features...")
        
        # Get port from environment or default to 5000
        port = int(os.environ.get('PORT', 5000))
        debug_mode = os.environ.get('FLASK_ENV') == 'development'
        
        app.run(debug=debug_mode, host='0.0.0.0', port=port)

    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"Error starting app: {str(e)}")
