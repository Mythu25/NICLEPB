import numpy as np
import pandas as pd
from collections import Counter
import re
import ast
from typing import Dict, List, Any
import time

class FeatureExtractor:
    def __init__(self):
        self.code_complexity_metrics = [
            'cyclomatic_complexity',
            'cognitive_complexity',
            'halstead_difficulty',
            'mccabe_complexity'
        ]
    
    def extract_session_features(self, session_data: Dict) -> Dict[str, float]:
        """Extract comprehensive features from session data"""
        features = {}
        
        # Time-based features
        total_time = session_data['start_time'] - time.time()
        features['total_session_time'] = total_time
        features['runs_per_minute'] = len(session_data['runs']) / (total_time / 60) if total_time > 0 else 0
        
        # Typing behavior
        keystrokes = session_data['keystrokes']
        if keystrokes:
            timestamps = [k['timestamp'] for k in keystrokes]
            typing_intervals = np.diff(timestamps)
            features['avg_typing_speed'] = 60 / np.mean(typing_intervals) if len(typing_intervals) > 0 else 0
            features['typing_variance'] = np.var(typing_intervals)
        
        # Code characteristics
        codes = session_data['code_history']
        if codes:
            code_lengths = [len(code) for code in codes]
            features['avg_code_length'] = np.mean(code_lengths)
            features['code_length_std'] = np.std(code_lengths)
            features['max_code_length'] = max(code_lengths)
        
        # Error patterns
        features['error_rate'] = len(session_data['errors']) / max(1, len(session_data['runs']))
        features['syntax_errors'] = sum(1 for e in session_data['errors'] if 'SyntaxError' in str(e))
        
        # Execution success
        successful_runs = sum(1 for r in session_data['runs'] if r.get('success', False))
        features['success_rate'] = successful_runs / max(1, len(session_data['runs']))
        
        # Code quality features
        if codes:
            code_features = self._extract_code_features(codes[-1] if codes else "")
            features.update(code_features)
        
        # Behavioral patterns
        features['fixation_ratio'] = self._calculate_fixation_ratio(session_data)
        features['pause_frequency'] = self._calculate_pause_frequency(keystrokes)
        
        return {k: float(v) for k, v in features.items()}
    
    def _extract_code_features(self, code: str) -> Dict[str, float]:
        """Extract static code features"""
        features = {}
        
        # Basic metrics
        features['lines_of_code'] = len([line for line in code.split('\n') if line.strip()])
        features['unique_tokens'] = len(set(re.findall(r'\w+', code)))
        
        # Complexity metrics
        try:
            tree = ast.parse(code)
            features['function_count'] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            features['class_count'] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            features['control_flow'] = sum(1 for node in ast.walk(tree) 
                                         if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)))
        except:
            features['function_count'] = 0
            features['class_count'] = 0
            features['control_flow'] = 0
        
        # Halstead metrics (simplified)
        operators = len(re.findall(r'[\+\-\*\/\$\$\$\$\{\}]', code))
        operands = len(re.findall(r'\w+', code))
        features['halstead_volume'] = operators * np.log2(operands) if operands > 0 else 0
        
        return features
    
    def _calculate_fixation_ratio(self, session_data: Dict) -> float:
        """Calculate fixation (pause) ratio in typing"""
        keystrokes = session_data['keystrokes']
        if len(keystrokes) < 2:
            return 0.0
        
        pauses = [ks['timestamp'] for ks in keystrokes[1:] if ks['timestamp'] - keystrokes[0]['timestamp'] > 1.0]
        return len(pauses) / len(keystrokes)
    
    def _calculate_pause_frequency(self, keystrokes: List) -> float:
        """Calculate frequency of pauses > 2 seconds"""
        if len(keystrokes) < 2:
            return 0.0
        
        timestamps = [ks['timestamp'] for ks in keystrokes]
        pauses = np.sum(np.diff(timestamps) > 2.0)
        return pauses / len(timestamps)