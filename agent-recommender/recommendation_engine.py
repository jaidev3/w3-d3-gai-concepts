from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import re
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spacy English model: python -m spacy download en_core_web_sm")
    nlp = None

@dataclass
class CodingAgent:
    name: str
    description: str
    strengths: List[str]
    weaknesses: List[str]
    best_for: List[str]
    languages: List[str]
    platforms: List[str]
    pricing: str
    features: List[str]
    integration: List[str]
    learning_curve: str  # "Easy", "Medium", "Hard"
    performance_score: float  # 1-10 scale

@dataclass
class TaskAnalysis:
    task_type: str
    complexity: str
    languages: List[str]
    domain: str
    requirements: List[str]
    urgency: str

@dataclass
class Recommendation:
    agent: CodingAgent
    score: float
    reasoning: str
    fit_percentage: int

class AgentKnowledgeBase:
    def __init__(self, db_file_path: str = "agents_db.json"):
        self.db_file_path = db_file_path
        self.agents = self._load_agents_from_json()
        
    def _load_agents_from_json(self) -> List[CodingAgent]:
        """Load agents from JSON database file"""
        try:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, self.db_file_path)
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            agents = []
            for agent_data in data.get('agents', []):
                agent = CodingAgent(
                    name=agent_data['name'],
                    description=agent_data['description'],
                    strengths=agent_data['strengths'],
                    weaknesses=agent_data['weaknesses'],
                    best_for=agent_data['best_for'],
                    languages=agent_data['languages'],
                    platforms=agent_data['platforms'],
                    pricing=agent_data['pricing'],
                    features=agent_data['features'],
                    integration=agent_data['integration'],
                    learning_curve=agent_data['learning_curve'],
                    performance_score=agent_data['performance_score']
                )
                agents.append(agent)
            
            return agents
            
        except FileNotFoundError:
            print(f"Warning: {self.db_file_path} not found. Using fallback agents.")
            return self._get_fallback_agents()
        except json.JSONDecodeError as e:
            print(f"Error parsing {self.db_file_path}: {e}. Using fallback agents.")
            return self._get_fallback_agents()
        except Exception as e:
            print(f"Error loading agents from {self.db_file_path}: {e}. Using fallback agents.")
            return self._get_fallback_agents()
    
    def _get_fallback_agents(self) -> List[CodingAgent]:
        """Fallback agents in case JSON loading fails"""
        return [
            CodingAgent(
                name="GitHub Copilot",
                description="AI pair programmer powered by OpenAI Codex",
                strengths=["Code completion", "Context awareness", "Multi-language support", "IDE integration"],
                weaknesses=["Subscription cost", "Privacy concerns", "Occasional incorrect suggestions"],
                best_for=["General development", "Code completion", "Learning new syntax", "Rapid prototyping"],
                languages=["Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Ruby", "PHP"],
                platforms=["VS Code", "Visual Studio", "Neovim", "JetBrains IDEs"],
                pricing="$10/month individual, $19/month business",
                features=["Real-time suggestions", "Docstring generation", "Test writing", "Code explanation"],
                integration=["Git", "GitHub", "Most popular IDEs"],
                learning_curve="Easy",
                performance_score=8.5
            ),
            CodingAgent(
                name="OpenAI ChatGPT",
                description="Conversational AI with strong coding capabilities and broad knowledge",
                strengths=["Versatile", "Strong explanation", "Multiple programming paradigms", "Creative solutions"],
                weaknesses=["No real-time integration", "Knowledge cutoff", "Hallucination risk"],
                best_for=["Learning", "Problem solving", "Code explanation", "Algorithm design"],
                languages=["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "Swift"],
                platforms=["Web interface", "API", "Mobile app"],
                pricing="Free tier, $20/month Plus",
                features=["Code generation", "Explanation", "Debugging help", "Documentation"],
                integration=["API", "Third-party extensions"],
                learning_curve="Easy",
                performance_score=8.3
            )
        ]
        
    def _initialize_agents(self) -> List[CodingAgent]:
        """Deprecated: Use _load_agents_from_json instead"""
        return self._load_agents_from_json()
    
    def get_all_agents(self) -> List[CodingAgent]:
        return self.agents
    
    def get_agent_by_name(self, name: str) -> Optional[CodingAgent]:
        for agent in self.agents:
            if agent.name.lower() == name.lower():
                return agent
        return None
    
    def reload_agents(self):
        """Reload agents from JSON file"""
        self.agents = self._load_agents_from_json()
    
    def save_agent_to_json(self, agent: CodingAgent):
        """Add a new agent to the JSON database"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, self.db_file_path)
            
            # Load existing data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add new agent
            agent_dict = asdict(agent)
            data['agents'].append(agent_dict)
            
            # Save back to file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Reload agents
            self.reload_agents()
            
        except Exception as e:
            print(f"Error saving agent to {self.db_file_path}: {e}")

class TaskAnalyzer:
    def __init__(self):
        self.complexity_keywords = {
            "simple": ["hello world", "basic", "simple", "tutorial", "learning", "beginner"],
            "medium": ["web app", "api", "database", "frontend", "backend", "integration"],
            "complex": ["machine learning", "ai", "distributed", "microservices", "enterprise", "architecture"]
        }
        
        self.domain_keywords = {
            "web": ["web", "website", "frontend", "backend", "html", "css", "react", "vue", "angular"],
            "mobile": ["mobile", "app", "android", "ios", "react native", "flutter"],
            "data": ["data", "analysis", "ml", "ai", "machine learning", "analytics", "visualization"],
            "desktop": ["desktop", "gui", "tkinter", "qt", "electron"],
            "devops": ["deployment", "ci/cd", "docker", "kubernetes", "aws", "cloud"],
            "game": ["game", "unity", "pygame", "graphics", "3d"],
            "api": ["api", "rest", "graphql", "microservices", "backend"]
        }
        
        self.language_keywords = {
            "python": ["python", "django", "flask", "fastapi", "pandas", "numpy"],
            "javascript": ["javascript", "js", "node", "react", "vue", "angular"],
            "typescript": ["typescript", "ts"],
            "java": ["java", "spring", "android"],
            "c++": ["c++", "cpp"],
            "c#": ["c#", "csharp", ".net"],
            "go": ["go", "golang"],
            "rust": ["rust"],
            "php": ["php", "laravel"],
            "ruby": ["ruby", "rails"]
        }
    
    def analyze_task(self, task_description: str) -> TaskAnalysis:
        task_lower = task_description.lower()
        
        # Analyze complexity
        complexity = "medium"  # default
        for level, keywords in self.complexity_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                complexity = level
                break
        
        # Analyze domain
        domain = "general"
        for dom, keywords in self.domain_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                domain = dom
                break
        
        # Detect languages
        languages = []
        for lang, keywords in self.language_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                languages.append(lang)
        
        if not languages:
            languages = ["python"]  # default
        
        # Extract requirements using basic NLP
        requirements = self._extract_requirements(task_description)
        
        # Determine urgency
        urgency_keywords = ["urgent", "asap", "quickly", "fast", "deadline"]
        urgency = "normal"
        if any(keyword in task_lower for keyword in urgency_keywords):
            urgency = "high"
        
        # Determine task type
        task_type = self._determine_task_type(task_lower)
        
        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            languages=languages,
            domain=domain,
            requirements=requirements,
            urgency=urgency
        )
    
    def _extract_requirements(self, text: str) -> List[str]:
        requirements = []
        
        # Common requirement patterns
        patterns = [
            r"need(?:s)? (?:to )?(\w+(?:\s+\w+)*)",
            r"should (\w+(?:\s+\w+)*)",
            r"must (\w+(?:\s+\w+)*)",
            r"require(?:s)? (\w+(?:\s+\w+)*)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            requirements.extend(matches)
        
        return requirements[:5]  # Limit to top 5
    
    def _determine_task_type(self, text: str) -> str:
        task_types = {
            "development": ["build", "create", "develop", "code", "implement"],
            "debugging": ["debug", "fix", "error", "bug", "issue"],
            "learning": ["learn", "understand", "tutorial", "explain"],
            "optimization": ["optimize", "improve", "performance", "speed"],
            "refactoring": ["refactor", "cleanup", "reorganize", "restructure"]
        }
        
        for task_type, keywords in task_types.items():
            if any(keyword in text for keyword in keywords):
                return task_type
        
        return "development"

class RecommendationEngine:
    def __init__(self, knowledge_base: AgentKnowledgeBase):
        self.kb = knowledge_base
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._prepare_agent_vectors()
    
    def _prepare_agent_vectors(self):
        # Prepare text representations of agents for similarity matching
        agent_texts = []
        for agent in self.kb.get_all_agents():
            text = f"{' '.join(agent.strengths)} {' '.join(agent.best_for)} {' '.join(agent.languages)} {agent.description}"
            agent_texts.append(text)
        
        self.agent_vectors = self.vectorizer.fit_transform(agent_texts)
    
    def recommend_agents(self, task_analysis: TaskAnalysis, top_k: int = 3) -> List[Recommendation]:
        agents = self.kb.get_all_agents()
        scored_agents = []
        
        for i, agent in enumerate(agents):
            score = self._calculate_agent_score(agent, task_analysis)
            reasoning = self._generate_reasoning(agent, task_analysis, score)
            fit_percentage = min(100, int(score * 10))
            
            recommendation = Recommendation(
                agent=agent,
                score=score,
                reasoning=reasoning,
                fit_percentage=fit_percentage
            )
            scored_agents.append(recommendation)
        
        # Sort by score and return top k
        scored_agents.sort(key=lambda x: x.score, reverse=True)
        return scored_agents[:top_k]
    
    def _calculate_agent_score(self, agent: CodingAgent, task: TaskAnalysis) -> float:
        score = 0.0
        
        # Base performance score (20% weight)
        score += agent.performance_score * 0.2
        
        # Language compatibility (25% weight)
        language_match = len(set(task.languages) & set([lang.lower() for lang in agent.languages]))
        language_score = min(1.0, language_match / len(task.languages)) if task.languages else 0.5
        score += language_score * 2.5
        
        # Complexity match (20% weight)
        complexity_scores = {"easy": 1.0, "medium": 0.8, "hard": 0.6}
        if task.complexity == "simple" and agent.learning_curve == "Easy":
            score += 2.0
        elif task.complexity == "medium":
            score += complexity_scores.get(agent.learning_curve.lower(), 0.5) * 2.0
        elif task.complexity == "complex" and agent.learning_curve in ["Medium", "Hard"]:
            score += 2.0
        else:
            score += 1.0
        
        # Domain/use case match (15% weight)
        domain_match = any(use_case.lower() in ' '.join(agent.best_for).lower() 
                          for use_case in [task.domain, task.task_type])
        if domain_match:
            score += 1.5
        
        # Feature relevance (10% weight)
        feature_relevance = self._calculate_feature_relevance(agent, task)
        score += feature_relevance * 1.0
        
        # Urgency consideration (10% weight)
        if task.urgency == "high":
            if agent.learning_curve == "Easy":
                score += 1.0
            elif agent.name in ["GitHub Copilot", "Cursor", "Replit AI", "v0 by Vercel", "Codeium"]:  # Fast setup
                score += 0.8
        else:
            score += 0.5
        
        return min(10.0, score)
    
    def _calculate_feature_relevance(self, agent: CodingAgent, task: TaskAnalysis) -> float:
        relevant_features = 0
        total_features = len(agent.features)
        
        feature_keywords = {
            "debugging": ["debug", "error", "test"],
            "learning": ["explanation", "documentation", "tutorial"],
            "collaboration": ["team", "share", "collaboration"],
            "deployment": ["deployment", "cloud", "production"]
        }
        
        task_keywords = feature_keywords.get(task.task_type, [])
        for feature in agent.features:
            if any(keyword in feature.lower() for keyword in task_keywords):
                relevant_features += 1
        
        return relevant_features / total_features if total_features > 0 else 0.0
    
    def _generate_reasoning(self, agent: CodingAgent, task: TaskAnalysis, score: float) -> str:
        reasons = []
        
        # Language support
        matching_languages = set(task.languages) & set([lang.lower() for lang in agent.languages])
        if matching_languages:
            reasons.append(f"Supports {', '.join(matching_languages)}")
        
        # Complexity match
        if task.complexity == "simple" and agent.learning_curve == "Easy":
            reasons.append("Easy to learn for simple tasks")
        elif task.complexity == "complex" and agent.performance_score > 8.0:
            reasons.append("High-performance tool suitable for complex projects")
        
        # Domain expertise
        domain_matches = [use_case for use_case in agent.best_for 
                         if task.domain.lower() in use_case.lower() or task.task_type.lower() in use_case.lower()]
        if domain_matches:
            reasons.append(f"Specialized for {', '.join(domain_matches[:2])}")
        
        # Key strengths
        if len(agent.strengths) > 0:
            reasons.append(f"Key strengths: {', '.join(agent.strengths[:2])}")
        
        if not reasons:
            reasons.append("Good general-purpose coding assistant")
        
        return ". ".join(reasons)
